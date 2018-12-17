from torch.utils.data import Dataset
import numpy as np
from transform import Transform
import cv2
import torch


class jdDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.COMPONETS = self.jdComponent()


    def __getitem__(self, index):
        pass

    def __len__(self):
        return 11393

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

        component = [facial_outer_contour, left_eyebrow, right_eyebrow, nose_bridge, nose_boundary]
        component += [left_upper_eyelid, right_upper_eyelid, left_lower_eyelid, right_lower_eyelid]
        component += [upper_upper_lip, upper_lower_lip, lower_upper_lip, lower_lower_lip]
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
        for i in range(13):
            maxVal = heatmap_[:, :, i].max()
            minVal = heatmap_[:, :, i].min()
            if maxVal == minVal:
                heatmap_[:, :, i] = 0
            else:
                heatmap_[:, :, i] = (heatmap_[:, :, i] - minVal) / (maxVal - minVal)
        return heatmap_
