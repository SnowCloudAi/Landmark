from torch.utils.data import Dataset
import numpy as np
from transform import Transform
import cv2
import torch
from glob import glob


class jdDataset(Dataset):

    def __init__(self):
        super().__init__()
        
        image_paths = sorted(glob('/home/ubuntu/FaceDatasets/jd/Training_data/*/picture/*.jpg'))
        landmarks_paths = sorted(glob('/home/ubuntu/FaceDatasets/jd/Training_data/*/landmark/*.txt'))

        for i in range(len(image_paths)):
            assert image_paths[i].split('/')[-1].split('.')[0] == landmarks_paths[i].split('/')[-1].split('.')[0], 'CHECK ERROR'

        self.image_paths = image_paths
        self.landmarks_paths = landmarks_paths
        assert len(self.image_paths) == len(self.landmarks_paths), 'CHECK ERROR'

        self.COMPONETS = self.jdComponent()
        self.transform = Transform(degrees=30, scale=(0.75, 1.0), translate=(-0.05, 0.05), colorjitter=0.3, out_size=256, flip=True)

    def __getitem__(self, index):
        """
            1. 从硬盘中用cv2读取BGR图片，转换成RGB公式
            2. 将landarmk的[196]转换成[92,2]的格式
            3. 送入T之后生成图片的tensor和numpy的landmark
            4. 制作boundary的heatmap（先resize再描点） np.where去除小于0的点，和大于256的点
        """
        img = cv2.imread(self.image_paths[index])
        landmark =  np.genfromtxt(self.landmarks_paths[index], skip_header=1).astype('float32') # float32 (nPoints, 2)

        img_tensor, landmark = self.transform(img, landmark)

        landmark_ = np.where(landmark > 0, landmark, 0.)
        landmark_ = np.where(landmark_ < 256, landmark_, 256.)

        heatmap = self.drawGaussian(landmark_)

        landmark = landmark.flatten()
        return img_tensor, torch.from_numpy(heatmap).float(), torch.from_numpy(landmark).float()

    def __len__(self):
        return len(self.image_paths)
    
    def jdComponent(self):
        """
            eyebrow: 眉毛
            eyelid: 眼皮
            nose boundary: 鼻梁
        """
        facial_outer_contour = [i for i in range(33)]
        left_eyebrow = [33 + i for i in range(9)] + [33, ]
        right_eyebrow = [42 + i for i in range(9)] + [42, ]

        nose_bridge = [51 + i for i in range(4)]
        nose_boundary = [57 + i for i in range(7)]
        left_upper_eyelid = [66 + i for i in range(5)]
        right_upper_eyelid = [75 + i for i in range(5)]
        left_lower_eyelid = [70 + i for i in range(4)] + [66, ]
        right_lower_eyelid = [79 + i for i in range(4)] + [75, ]

        upper_upper_lip = [84 + i for i in range(7)]
        upper_lower_lip = [96 + i for i in range(5)]
        lower_upper_lip = [100 + i for i in range(4)] + [96, ]
        lower_lower_lip = [90 + i for i in range(6)] + [84, ]

        component = [facial_outer_contour, lower_lower_lip, lower_upper_lip, upper_lower_lip, upper_upper_lip]
        component += [nose_boundary, nose_bridge, left_lower_eyelid, left_upper_eyelid, left_eyebrow]
        component += [right_lower_eyelid, right_upper_eyelid, right_eyebrow]
        return component

    def drawGaussian(self, landmark):
        landmark = landmark / 4
        heatmap = np.zeros((len(self.COMPONETS), 64, 64), dtype=np.uint8)
        sigma = 1
        num = 3.0
        for i in range(len(self.COMPONETS)):
            for j in range(len(self.COMPONETS[i]) - 1):
                p1 = self.COMPONETS[i][j]
                p2 = self.COMPONETS[i][j+1]
                heatmap[i] = cv2.line(heatmap[i], (int(landmark[p1, 0]), int(landmark[p1, 1])),
                 (int(landmark[p2, 0]), int(landmark[p2, 1])), (255), 1, )

            heatmap[i] = cv2.distanceTransform(255 - heatmap[i], cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        heatmap_ = np.exp(-1.0 * heatmap ** 2 / (2.0 * sigma ** 2))
        heatmap_ = np.where(heatmap < (num * sigma), heatmap_, 0)

        return heatmap_
