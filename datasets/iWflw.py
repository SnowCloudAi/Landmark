import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

# ia.seed(1)

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
import time

from random import shuffle

class iWflw(Dataset):
    def __init__(self, mode='train', transform_mode='train', attr='all', balance='False', value='1'):
        super().__init__()
        
        self.mode = mode
        self.transform_mode = transform_mode
        
        train_path = '../data/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt'
        test_path = '../data/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt'
        path = train_path if self.mode == 'train' else test_path
        
        attr_list = ['pose', 'expression', 'illumination', 'make_up', 'occlusion', 'blur', 'all']
        assert attr in attr_list, 'attribute %s not an option' % attr
        self.attr_pos = attr_list.index(attr) + 2
        
        with open(path, 'r') as f:
            annotations = f.readlines()
        
        labels_positive = []
        labels_negative = []
        for anno in annotations:
            label_ = self.get_anno(anno)
            if attr != 'all':
                if label_[self.attr_pos] == value:
                    labels_positive.append(label_)
                else:
                    labels_negative.append(label_)
            else:
                labels_positive.append(label_)
        if attr != 'all':            
            shuffle(labels_positive); shuffle(labels_negative)
            if len(labels_positive) > len(labels_negative):
                labels_negative = labels_negative * 100
                labels = labels_positive+labels_negative[:len(labels_positive)]
            else: 
                labels_positive_ = labels_positive * 100
                labels = labels_negative+labels_positive_[:len(labels_negative)]
            shuffle(labels)
        
        if balance == 'True':
            self.labels = labels
        else:
            self.labels = labels_positive
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.components = [[i for i in range(33)], [76, 87, 86, 85, 84, 83, 82], [88, 95, 94, 93, 92], [88, 89, 90, 91, 92],
              [76, 77, 78, 79, 80, 81, 82], [55 + i for i in range(5)], [51 + i for i in range(4)],
              [60, 67, 66, 65, 64], [60 + i for i in range(5)], [33 + i for i in range(9)], [68, 75, 74, 73, 72],
              [68 + i for i in range(5)], [42 + i for i in range(9)]]
    
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        
        landmarks, bbox, pose, expression, illumination, make_up, occlusion, blur, image_name = self.labels[index]
        bbox = self.refine_bbox(bbox, landmarks)
        
        path = '../data/WFLW/WFLW_images/%s' % image_name
        bgr = cv2.imread(path)
        
        transformed_img, transformed_lands = self.itransform(bgr, bbox, landmarks, pad_rate=0.10, scale=256, transform_mode=self.transform_mode)
        transformed_lands = self.key_to_list(transformed_lands)
        
        img_tensor = self.transform(transformed_img)
        
        landmark = np.zeros((98, 2))
        landmark[:, 0] = transformed_lands[0:98*2:2]
        landmark[:, 1] = transformed_lands[1:98*2:2]

        landmark_ = np.where(landmark > 0, landmark, 0.)
        landmark_ = np.where(landmark_ < 256, landmark_, 256.)
        
        heatmap = self.drawGaussian(landmark_)
        landmark = landmark.reshape(196)
        
        return img_tensor, torch.from_numpy(heatmap).float(), torch.from_numpy(landmark).float()
        
    def get_anno(self, anno):
        """从一行标注里获取并返回landmarks, bbox, image_name"""
        anno = anno.split()
        
        landmarks = anno[:196]
        landmarks = [eval(e) for e in landmarks]
        bbox = anno[196:200]
        bbox = [eval(e) for e in bbox]

        pose, expression, illumination, make_up, occlusion, blur, image_name = anno[-7:]
        
        return landmarks, bbox, pose, expression, illumination, make_up, occlusion, blur, image_name
    
    def rectangle_to_square_v2(self, bbox):
        """修正bbox长方形为正方形，若触及边界不用修正，留待padding"""
        x_len = abs(bbox[0]-bbox[2])
        y_len = abs(bbox[1]-bbox[3])
        max_len = max(x_len, y_len)

        center_x = (bbox[0]+bbox[2]) / 2.
        center_y = (bbox[1]+bbox[3]) / 2.

        return center_x-max_len/2., center_y-max_len/2., center_x+max_len/2., center_y+max_len/2.

    def refine_bbox(self, bbox, landmarks):
        """
        输入原始bbox和landmarks，输出修正过的bbox
        修正方法：求出landmarks的范围，和原始bbox取并集，并取正方形
        """
        land_x = [landmarks[i] for i in range(0, len(landmarks), 2)]
        land_y = [landmarks[i+1] for i in range(0, len(landmarks), 2)]

        land_bbox = [min(land_x), min(land_y), max(land_x), max(land_y)]

        refined_bbox = [min(bbox[0], land_bbox[0]), min(bbox[1], land_bbox[1]), max(bbox[2], land_bbox[2]), max(bbox[3], land_bbox[3])]
        refined_bbox = self.rectangle_to_square_v2(refined_bbox)

        return refined_bbox
    
    def itransform(self, bgr, bbox, landmarks, pad_rate=0.2, scale=256, transform_mode='train'):
        """变换图片，获取crop人脸及变换后的关键点"""
        left = int(bbox[0])
        top = int(bbox[1])
        right = int(bgr.shape[1] - bbox[2])
        bottom = int(bgr.shape[0] - bbox[3])
        
        if transform_mode == 'train':
            pad = int(pad_rate*abs(bbox[0]-bbox[2])/2)
            # 定义一个变换序列，可扩展
            seq = iaa.Sequential([
                iaa.CropAndPad(px=(-(top-pad), -(right-pad), -(bottom-pad), -(left-pad))), 
                iaa.CropAndPad(percent=([-0.10, 0.10], [-0.10, 0.10], [-0.10, 0.10], [-0.10, 0.10]), pad_mode=["constant", "edge"]), 
                # iaa.Fliplr(0.5), 
                iaa.GaussianBlur(sigma=1.0), 
                iaa.Affine(
                    # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-30, 30),
                #     shear=(-5, 5)
                ), 
                iaa.Scale(scale), 
            ])
        elif transform_mode == 'soft':
            pad = int(pad_rate*abs(bbox[0]-bbox[2])/2)
            seq = iaa.Sequential([
                iaa.CropAndPad(px=(-(top-pad), -(right-pad), -(bottom-pad), -(left-pad))), 
                # iaa.GaussianBlur(sigma=(0, 3.0)), 
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, 
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-15, 15), 
                    shear=(-5, 5), 
                ), 
                iaa.Scale((0.25, 0.40)), 
                iaa.Scale(scale), 
            ])
        
        else:
            pad = 0
            seq = iaa.Sequential([
                iaa.CropAndPad(px=(-(top-pad), -(right-pad), -(bottom-pad), -(left-pad))), 
                iaa.Scale(scale), 
            ])
        
        seq_det = seq.to_deterministic()
        
        keypoint_lst = [ia.Keypoint(x=landmarks[i], y=landmarks[i+1]) for i in range(0, len(landmarks), 2)]
        keypoints=ia.KeypointsOnImage(keypoint_lst, shape=bgr.shape)
        
        image_aug = seq_det.augment_images([bgr])[0] 
        keypoints_aug = seq_det.augment_keypoints([keypoints])[0]
        
        return image_aug, keypoints_aug
    
    def key_to_list(self, keypoint):
        """把imgaug的keypoint转为list并返回"""
        result = np.ravel(np.round(keypoint.get_coords_array())).astype(int).tolist()
        return result
    
    def drawGaussian(self, landmark):
        components = self.components
        landmark = landmark / 4
        heatmap = np.zeros((len(components), 64, 64), dtype=np.uint8)
        sigma = 1
        num = 3.0
        for i in range(len(components)):
            for j in range(len(components[i]) - 1):
                p1 = components[i][j]
                p2 = components[i][j+1]
                heatmap[i] = cv2.line(heatmap[i], (int(landmark[p1, 0]), int(landmark[p1, 1])),
                 (int(landmark[p2, 0]), int(landmark[p2, 1])), (255), 1, )

            heatmap[i] = cv2.distanceTransform(255 - heatmap[i], cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        heatmap_ = np.exp(-1.0 * heatmap ** 2 / (2.0 * sigma ** 2))
        heatmap_ = np.where(heatmap < (num * sigma), heatmap_, 0)
        
        return heatmap_

