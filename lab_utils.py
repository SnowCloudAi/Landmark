import cv2
from IPython.display import display, Image
import matplotlib.pyplot as plt
import numpy as np


def lrSchedule(base_lr, iter, iters, epoch=0, time=0, step=(30, 60, 90), target_lr=0.0, mode='cosine', per_epoch=0):
    lr = target_lr if target_lr else base_lr
    iters = iters if iter < iters else iter

    # every iteration
    if mode == 'cosine':
        lr += (base_lr - target_lr) * (1 + np.cos(np.pi * iter / iters)) / 2.0
    # every epochs
    elif mode == 'step':
        if epoch in step:
            pass
    elif mode == 'cycleCosine':
        time = time if time else 4
        T = iters // time
        cur_iter, cur_time = iter % T, iter // T
        cur_base = (base_lr - target_lr) / time * (time - cur_time) + target_lr
        lr += (cur_base - target_lr) * (1 + np.cos(np.pi * cur_iter / T)) * 0.5
    elif mode == 'hhh':
        if epoch < 200:
            cur_lr = base_lr
            cur_iter = iter
            lr += (cur_lr - target_lr) * (1 + np.cos(np.pi * cur_iter / 200 * per_epoch)) * 0.5
        elif epoch < 300:
            cur_lr = (base_lr - target_lr) / 4 * 3 + target_lr
            cur_iter = iter - per_epoch * 200
            lr += (cur_lr - target_lr) * (1 + np.cos(np.pi * cur_iter / 100 * per_epoch)) * 0.5

        elif epoch < 350:
            cur_lr = (base_lr - target_lr) / 4 * 2 + target_lr
            cur_iter = iter - per_epoch * 300
            lr += (cur_lr - target_lr) * (1 + np.cos(np.pi * cur_iter / 50 * per_epoch)) * 0.5

        else:
            cur_lr = (base_lr - target_lr) / 4 + target_lr
            cur_iter = iter - epoch * 350
            lr += (cur_lr - target_lr) * (1 + np.cos(np.pi * cur_iter / 50 * per_epoch)) * 0.5

    return lr


components = [[i for i in range(33)], [76, 87, 86, 85, 84, 83, 82], [88, 95, 94, 93, 92], [88, 89, 90, 91, 92],
              [76, 77, 78, 79, 80, 81, 82], [55 + i for i in range(5)], [51 + i for i in range(4)],
              [60, 67, 66, 65, 64], [60 + i for i in range(5)], [33 + i for i in range(9)], [68, 75, 74, 73, 72],
              [68 + i for i in range(5)], [42 + i for i in range(9)]]


def getBBox(points):
    """

    :param points: x1, y1, x2, y2
    :return:
    """

    bbox = np.array([min(points[0::4]), min((points[1::4])), max(points[2::4]), max(points[3::4])])
    return bbox


def enlargeBBox(bbox, factor=0.05):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    x1 = x1 - width * factor
    x2 = x2 + width * factor
    y1 = y1 - height * factor
    y2 = y2 + height * factor

    return np.array([x1, y1, x2, y2])


def show(img_path):
    display(Image(img_path))


def landmark(path, points, detection=False):
    img = cv2.imread('WFLW_crop/' + path)
    if detection:
        x0, y0, x1, y1 = points[98 * 2:98 * 2 + 4]
        # img = img[y0:y1, x0:x1]
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
    for i in range(0, 98 * 2, 2):
        cv2.circle(img, (points[i], points[i + 1]), 1, (23, 25, 0), 1)
    plt.imshow(img[:, :, ::-1])


def drawLine(img, points, color=(25, 100, 0), return_image=True, thickness=1):
    """

    :param thickness:
    :param return_image:
    :param img:
    :param points: index 9 and 12 must be close (196 landmarks and 4 detection boxes)
    :param color:
    :return:
    """

    if return_image:
        color = color if len(img.shape) else (255,)
        for com in range(len(components)):
            for i in range(len(components[com]) - 1):
                p1 = components[com][i]
                p2 = components[com][i + 1]
                cv2.line(img, (points[p1 * 2], points[p1 * 2 + 1]),
                         (points[p2 * 2], points[p2 * 2 + 1]), color, thickness, )
            if com == 9 or com == 12:
                p1 = components[com][0]
                p2 = components[com][-1]
                cv2.line(img, (points[p1 * 2], points[p1 * 2 + 1]),
                         (points[p2 * 2], points[p2 * 2 + 1]), color, thickness, )
    else:
        img = np.zeros((img.shape[0], img.shape[1], 13), dtype=np.uint8)
        color = (255,)
        for com in range(len(components)):
            image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            for i in range(len(components[com]) - 1):
                p1 = components[com][i]
                p2 = components[com][i + 1]
                cv2.line(image, (points[p1 * 2], points[p1 * 2 + 1]),
                         (points[p2 * 2], points[p2 * 2 + 1]),
                         color,
                         2, )
            if com in [9, 12]:
                p1 = components[com][0]
                p2 = components[com][-1]
                cv2.line(image, (points[p1 * 2], points[p1 * 2 + 1]),
                         (points[p2 * 2], points[p2 * 2 + 1]), color, thickness, )
            img[:, :, com] = image
    return img


def drawDistanceImg(img, points, color=(25, 100, 0)):
    img = drawLine(img, points, color=color, return_image=False)
    assert img.shape[2] == 13, "This is not the 13 components heatmap!"
    for i in range(13):
        img[:, :, i] = cv2.distanceTransform(255 - img[:, :, i], cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return img


def drawGaussianHeatmap(img, points, color=(25, 100, 0), sigma=4):
    dist_img = drawDistanceImg(img, points, color=color)
    # heatmap = (1.0 / np.sqrt(2 * np.pi * sigma)) * np.exp(-1.0 * dist_img ** 2 / (2.0 * sigma ** 2))
    heatmap = np.exp(-1.0 * dist_img ** 2 / (2.0 * sigma ** 2))
    heatmap = np.where(dist_img < (3.0 * sigma), heatmap, 0)
    for i in range(13):
        maxVal = heatmap[:, :, i].max()
        minVal = heatmap[:, :, i].min()
        if maxVal == minVal:
            heatmap[:, :, i] = 0
        else:
            heatmap[:, :, i] = (heatmap[:, :, i] - minVal) / (maxVal - minVal)
    return heatmap


def drawPoint(img, points, color=(25, 100, 0)):
    """

    :param color: 
    :param img: RGB Image
    :param points: list type
    :return:
    """
    for i in range(0, 98 * 2, 2):
        cv2.circle(img, (points[i], points[i + 1]), 2, color, 2)
    return img


def figPicture(data, heatmap, widehatHeatmap, pred_real_landmarks, pred_fake_landmarks):
    """

    :param data: [3, 256, 256] RGB
    :param heatmap: [13, 64, 64]
    :param widehatHeatmap: [13, 64, 64]
    :return: [64 * 4, 64 * 7, 1]
    """
    data = np.mean(cv2.resize(np.moveaxis(data, 0, 2), (64, 64)), axis=2, keepdims=False, dtype=np.float32)

    for com in range(len(components)):
        for i in range(len(components[com])):
            p1 = components[com][i]
            heatmap[com] = cv2.circle(heatmap[com], (pred_real_landmarks[p1 * 2], pred_real_landmarks[p1 * 2 + 1]), 2,
                                      (1), 2, )
            widehatHeatmap[com] = cv2.circle(widehatHeatmap[com],
                                             (pred_fake_landmarks[p1 * 2], pred_fake_landmarks[p1 * 2 + 1]), 2, (1),
                                             2, )

    heatmap = np.moveaxis(heatmap, 0, 2).copy()
    widehatHeatmap = np.moveaxis(widehatHeatmap, 0, 2).copy()

    line1 = np.concatenate([data] + [heatmap[..., i] for i in range(6)], axis=1)
    line2 = np.concatenate([heatmap[..., i + 6] for i in range(7)], axis=1)
    line3 = np.concatenate([data] + [widehatHeatmap[..., i] for i in range(6)], axis=1)
    line4 = np.concatenate([widehatHeatmap[..., i + 6] for i in range(7)], axis=1)

    fig = np.concatenate([line1, line2, line3, line4], axis=0)

    return fig * 256
