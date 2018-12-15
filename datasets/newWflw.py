from torch.utils.data import Dataset
import numpy as np
from transform import Transform
from lab_utils import components
import cv2
import torch
class newWflw(Dataset):

    def __init__(self, mode='train', ):
        super().__init__()
        train_path = '/home/ubuntu/FaceDatasets/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt'
        test_path = '/home/ubuntu/FaceDatasets/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt'
        path = train_path if mode is 'train' else test_path
        # Load landmarks as numpy ()
        self.landmarks = np.genfromtxt(path)[:, :98 * 2]
        # Load img and its boundaries heatmap
        self.img_paths = []
        with open(path, 'r') as file:
            for _ in range(self.landmarks.shape[0]):
                info = file.readline().split(' ')
                self.img_paths.append(info[-1][:-1])

        self.transform = Transform(degrees=30, scale=(0.75, 1.0), translate=(-0.05, 0.05), colorjitter=0.3, out_size=256, flip=True)

    def __len__(self):
        return self.landmarks.shape[0]

    def __getitem__(self, index):
        """
            1. 从硬盘中用cv2读取BGR图片，转换成RGB公式
            2. 将landarmk的[196]转换成[92,2]的格式
            3. 送入T之后生成图片的tensor和numpy的landmark
            4. 制作boundary的heatmap（先resize再描点） np.where去除小于0的点，和大于256的点
        """
        img = cv2.imread('/home/ubuntu/FaceDatasets/WFLW/WFLW_images/' + self.img_paths[index])

        landmark = np.zeros((98, 2))
        landmark[:, 0] = self.landmarks[index, 0:98*2:2]
        landmark[:, 1] = self.landmarks[index, 1:98*2:2]

        img_tensor, landmark = self.transform(img, landmark)

        landmark_ = np.where(landmark > 0, landmark, 0.)
        landmark_ = np.where(landmark_ < 256, landmark_, 256.)
        heatmap = self.drawGaussian(landmark_)

        # return img_tensor, torch.from_numpy(heatmap).float(), torch.from_numpy(self.landmarks[index]).float()
        return img_tensor, torch.from_numpy(heatmap).float(), torch.from_numpy(landmark_).float()


    def drawGaussian(self, landmark):
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
            # if i in [9, 12]:
            #     p1 = components[i][0]
            #     p2 = components[i][1]
            #     heatmap[i] = cv2.line(heatmap[i], (int(landmark[p1, 0]), int(landmark[p1, 1])),
            #                           (int(landmark[p2, 0]), int(landmark[p2, 1])), (255), 1, )

            heatmap[i] = cv2.distanceTransform(255 - heatmap[i], cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        heatmap_ = np.exp(-1.0 * heatmap ** 2 / (2.0 * sigma ** 2))
        heatmap_ = np.where(heatmap < (num * sigma), heatmap_, 0)
        for i in range(13):
            maxVal = heatmap_[:, :, i].max()
            minVal = heatmap_[:, :, i].min()
            if maxVal == minVal:
                heatmap_[:, :, i] = 0
            else:
                heatmap_[:, :, i] = (heatmap_[:, :, i] - minVal) / (maxVal - minVal)
        return heatmap_


