from torch.utils.data import Dataset
import torch
import sys
import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
from PIL import Image

class WFLW(Dataset):
    """
        return : torch.tensor(img, dtype=torch.float32), torch.tensor(heatmap_label, dtype=torch.float32, )
    """

    def __init__(self, mode='train', path=None, dataChannel=1):
        super().__init__()
        train_path = '/home/ubuntu/FaceDatasets/WFLW/WFLW_crop_annotations/list_98pt_rect_attr_train_test/resplit_0.1_train.txt'
        test_path = '/home/ubuntu/FaceDatasets/WFLW/WFLW_crop_annotations/list_98pt_rect_attr_train_test/resplit_0.1_test.txt'
        path = path if path else train_path if mode is 'train' else test_path
        # Load landmarks as numpy ()
        self.landmarks = np.genfromtxt(path)
        # Load img and its boundaries heatmap
        self.imgs = []
        self.heatmaps = []
        self.dataChannel = dataChannel
        with open(path, 'r') as file:
            for _ in range(self.landmarks.shape[0]):
                info = file.readline().split(' ')
                self.imgs.append(np.load(info[-2]))
                self.heatmaps.append(np.load(info[-1][:-1]))

    def __len__(self):
        return self.landmarks.shape[0]

    def __getitem__(self, index):
        img, heatmap, landmarks = np.moveaxis(self.imgs[index], 0, 2), self.heatmaps[index], self.landmarks[index,
                                                                                             :98 * 2]
        # -----------(Only Image )-----------
        # img = self.preProcessImage(img)
        # -----------(Including  Landmarks )-----------
        # img, heatmap, landmarks = self.flip(img, heatmap, landmarks)
        img = np.moveaxis(img, 2, 0).copy()

        return torch.tensor(img, dtype=torch.float32), torch.tensor(heatmap.copy(),
                                                                    dtype=torch.float32), torch.FloatTensor(
            list(landmarks))

    def preProcessImage(self, img, SNR=0.95):
        aug_choice = np.random.randint(low=0, high=2)
        if aug_choice is 0:
            # 1. Gaussian Blur
            kernel_size = np.random.randint(low=1, high=8) // 2 * 2 + 1
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)[:, :, np.newaxis]
        elif aug_choice is 1:
            # 2. Salt Pepper
            mask = np.random.choice((0, 1, 2,), size=(img.shape[0], img.shape[1], 1),
                                    p=[SNR, (1 - SNR) / 2, (1 - SNR) / 2])
            img[mask == 1] = 255  # Salt
            img[mask == 2] = 0  # Pepper
        elif aug_choice is 2:
            # 3. Random 遮挡
            stride = np.random.randint(low=5, high=20)
            choice = np.random.randint(low=0, high=2)
            if choice == 0:
                img[::stride] = 0
            else:
                img[:, ::stride] = 0
        # 4. Invert Pixel
        return img

    def flip(self, img, heatmap, landmarks):
        choice = np.random.randint(low=0, high=3)
        # if horizontal:
        if choice is 0:
            img = img[:, ::-1, :]
            heatmap = heatmap[:, :, ::-1]
            landmarks[0::2] = 256 - landmarks[0::2] - 1
        elif choice is 1:
            img = img[::-1, :, :]
            heatmap = heatmap[:, ::-1]
            landmarks[1::2] = 256 - landmarks[1::2] - 1
        return img, heatmap, landmarks

    def rotate(self, img, heatmap, landmarks, theta):
        theta = theta * np.pi / 180
        rotateMatrix = np.array([[np.sin]])
        img[:, :, 0] = cv2.warpAffine(img[:, :, 0], )
        return img, heatmap, landmarks
