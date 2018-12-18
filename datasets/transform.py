import cv2
import numpy as np
import warnings
import numbers
import random
import math
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

match_parts_68 = np.array([[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],  # outline
                           [17, 26], [18, 25], [19, 24], [20, 23], [21, 22],  # eyebrow
                           [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],  # eye
                           [31, 35], [32, 34],  # nose
                           [48, 54], [49, 53], [50, 52], [59, 55], [58, 56],  # outer mouth
                           [60, 64], [61, 63], [67, 65]])
match_parts_98 = np.array(
    [[0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27], [6, 26], [7, 25], [8, 24], [9, 23], [10, 22], [11, 21],
     [12, 20], [13, 19], [14, 18], [15, 17],  # outline
     [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [41, 47], [40, 48], [39, 49], [38, 50],  # eyebrow
     [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [67, 73], [66, 74], [65, 75], [96, 97],  # eye
     [55, 59], [56, 58],  # nose
     [76, 82], [77, 81], [78, 80], [87, 83], [86, 84],  # outer mouth
     [88, 92], [89, 91], [95, 93]])

match_parts_106 = np.array(
    [[0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27], [6, 26], [7, 25], [8, 24], [9, 23], [10, 22],
     [11, 21],
     [12, 20], [13, 19], [14, 18], [15, 17],  # outline
     [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [41, 47], [40, 48], [39, 49], [38, 50],  # eyebrow
     [55, 65], [66, 79], [67, 78], [68, 77], [69, 76], [70, 75], [71, 82], [72, 81], [73, 80], [74, 83],
     [104, 105], [56, 64], [57, 63], [58, 62], [59, 61], [84, 90], [85, 89], [86, 88], [96, 100], [97, 99],
     [95, 91], [94, 92], [103, 102]])


def rotatepoints(landmarks, center, rot):
    center_coord = np.zeros_like(landmarks)
    center_coord[:, 0] = center[0]
    center_coord[:, 1] = center[1]
    angle = math.radians(rot)
    rot_matrix = np.array([[math.cos(angle), -1 * math.sin(angle)],
                           [math.sin(angle), math.cos(angle)]])
    rotate_coords = np.dot((landmarks - center_coord), rot_matrix) + center_coord
    return rotate_coords


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Code from: https://stackoverflow.com/a/18927641.
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.
        d, Z, [tform] = procrustes(X, Y)
    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.
    scaling
        if False, the scaling component of the transformation is forced
        to 1
    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.
    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()
    Z
        the matrix of transformed Y-values
    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y
    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform


class Transform(object):
    """Data augmentation code for face alignment.
       degrees: a positive integer, a tuple or list, optimal value: 30 or (-30,30)
       scale: a tuple or list, optimal value: (0.75, 1.0)
       translate: a tuple or list, optimal value: (-0.1, 0.1)
       colorjitter:  a positive number, optimal value: 0.3
       out_size: a positive integer, a tuple or list
       flip: flip image and landmarks"""

    def __init__(self, degrees, scale, translate, colorjitter, out_size, flip=True):
        if isinstance(degrees, numbers.Number):
            if degrees < 0 or not isinstance(degrees, int):
                raise ValueError("If degrees is a single number, it must be a positive integer.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            for degree in degrees:
                if not isinstance(degree, int):
                    raise ValueError("degrees values should be integers.")
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (-1.0 <= t <= 1.0):
                    raise ValueError("translation values should be between -1 and 1.")
                if t > 0.1 or t < - 0.1:
                    warnings.warn("The optimal translation values should be between -0.1 and 0.1.")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive.")
        self.scale = scale

        if colorjitter is not None:
            if isinstance(colorjitter, numbers.Number):
                if colorjitter < 0:
                    raise ValueError("If colorjitter is a single number, it must be positive.")
                if colorjitter >= 1:
                    raise ValueError("If colorjitter is a single number, it must less than 1.")
                if colorjitter >= 0.5:
                    warnings.warn("color jitter values should not exceed 0.5.")
                self.colorjitter = (1 - colorjitter, 1 + colorjitter)
            else:
                raise ValueError("color jitter should be a single number.")

        if isinstance(out_size, numbers.Number):
            if out_size < 0:
                raise ValueError("If out_size is a single number, it must be positive.")
            if not isinstance(out_size, int):
                raise ValueError("If out_size is a single number, it must be a positive integer.")
            self.out_size = (out_size, out_size)
        else:
            assert isinstance(out_size, (tuple, list)) and len(out_size) == 2, \
                "out_size should be a list or tuple and it must be of length 2."
            for size in out_size:
                if not isinstance(size, int):
                    raise ValueError("out_size values should be integers.")
            self.out_size = out_size
        if not isinstance(flip, bool):
            raise ValueError("Flip should be boolean, but got {}.".format(type(flip)))

        self.flip = flip
        self.to_tensor = ToTensor()

    def flippoints(self, landmarks, width):
        nPoints = landmarks.shape[0]
        assert nPoints in (68, 98, 106), 'flip {} nPoints is not supported'
        if nPoints == 98:
            pairs = match_parts_98
        elif nPoints == 68:
            pairs = match_parts_68
        else:
            parirs = match_parts_106
        flandmarks = landmarks.copy()

        for pair in pairs:
            flandmarks[pair[0]] = landmarks[pair[1]]
            flandmarks[pair[1]] = landmarks[pair[0]]
        flandmarks[:, 0] = width - flandmarks[:, 0] - 1

        return flandmarks

    def __call__(self, image, landmarks):
        h, w, c = image.shape
        npoints, dim = landmarks.shape
        if npoints not in [68, 98]:
            raise Exception('For now, transform only support dataset with 68/98 landmarks.')
        if dim != 2:
            raise Exception('Landmarks size should be n*2, but got {}*{}.'.format(npoints, dim))

        """1.rotate"""
        rot = random.randint(self.degrees[0], self.degrees[1])
        r_landmarks = rotatepoints(landmarks, [w / 2, h / 2], rot)
        left = min(r_landmarks[:, 0])
        right = max(r_landmarks[:, 0])
        top = min(r_landmarks[:, 1])
        bot = max(r_landmarks[:, 1])
        r_landmarks -= [left, top]
        r_landmarks *= [self.out_size[0] / (right - left), self.out_size[1] / (bot - top)]

        """2.scale"""
        sx = random.uniform(self.scale[1], self.scale[0])
        sy = sx * random.uniform(0.8, 1.2)
        s_landmarks = r_landmarks * [sx, sy]
        dx = (1 - sx) * self.out_size[0] * 0.5
        dy = (1 - sy) * self.out_size[1] * 0.5

        """3.translation"""
        dx = random.uniform(self.translate[0], self.translate[1]) * self.out_size[0]
        dy = random.uniform(self.translate[0], self.translate[1]) * self.out_size[1]
        t_landmarks = s_landmarks + [dx, dy]

        """4.procrustes analysis"""
        d, Z, tform = procrustes(t_landmarks, landmarks)
        M = np.zeros([2, 3], dtype=np.float32)
        M[:2, :2] = tform['rotation'].T * tform['scale']
        M[:, 2] = tform['translation']
        img = cv2.warpAffine(image, M, (self.out_size[1], self.out_size[0]))
        w_landmarks = np.dot(landmarks, tform['rotation']) * tform['scale'] + tform['translation']

        """5.flip"""
        if self.flip and random.random() > 0.5:
            img = img[:, ::-1]
            w_landmarks = self.flippoints(w_landmarks, self.out_size[0])
        # show_image(img, w_landmarks
        """6.color jitter"""
        img = self.to_tensor(img.copy())
        img[0, :, :].mul_(random.uniform(self.colorjitter[0], self.colorjitter[1])).clamp_(0, 1)
        img[1, :, :].mul_(random.uniform(self.colorjitter[0], self.colorjitter[1])).clamp_(0, 1)
        img[2, :, :].mul_(random.uniform(self.colorjitter[0], self.colorjitter[1])).clamp_(0, 1)

        return img, w_landmarks

# t = (degrees=30, scale=(0.75,1.0), translate=(-0.05, 0.05), colorjitter=0.3, out_size=128, flip=True)
# def loadFromPts(filename):
#     landmarks = np.genfromtxt(filename, skip_header=3, skip_footer=1)
#     landmarks = landmarks - 1
#     return landmarks
# def show_image(image, landmarks, box=None):
#     fig = plt.figure(figsize=plt.figaspect(.5))
#     ax = fig.add_subplot(1, 1, 1)
#     ax.imshow(image)
#     num_points = landmarks.shape[0]
#     if num_points == 68:
#         ax.plot(landmarks[0:17,0],landmarks[0:17,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[17:22,0],landmarks[17:22,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[22:27,0],landmarks[22:27,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[27:31,0],landmarks[27:31,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[31:36,0],landmarks[31:36,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[36:42,0],landmarks[36:42,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[42:48,0],landmarks[42:48,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[48:60,0],landmarks[48:60,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[60:68,0],landmarks[60:68,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#     elif num_points == 98:
#         ax.plot(landmarks[0:33,0],landmarks[0:33,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[33:38,0],landmarks[33:38,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[37:42,0],landmarks[37:42,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[42:46,0],landmarks[42:46,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[45:51,0],landmarks[45:51,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[51:55,0],landmarks[51:55,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[55:60,0],landmarks[55:60,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[60:65,0],landmarks[60:65,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[64:68,0],landmarks[64:68,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[68:73,0],landmarks[68:73,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[72:76,0],landmarks[72:76,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[76:83,0],landmarks[76:83,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[82:88,0],landmarks[82:88,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[88:93,0],landmarks[88:93,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[92:96,0],landmarks[92:96,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[96,0],landmarks[96,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#         ax.plot(landmarks[97,0],landmarks[97,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
#     # if box is not None:
#     #     currentAxis=plt.gca()
#     #     box = enlarge_box(box,0.05)
#     #     xmin, ymin, xmax, ymax = box
#     #     rect=patches.Rectangle((xmin, ymin),xmax-xmin,ymax-ymin,linewidth=2,edgecolor='r',facecolor='none')
#     #     currentAxis.add_patch(rect)
#     ax.axis('off')
#     plt.show()
# import skimage.io as io
#
# imgpath = '/home/xiang/pytorch/Shufflent-fa/Data/train/1.jpg'
# image = io.imread(imgpath)
# landmarks = loadFromPts(imgpath[:-3] + 'pts')
# show_image(image, landmarks)
# img, landmarks = t(image, landmarks)
