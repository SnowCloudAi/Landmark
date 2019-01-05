import torch
import numpy as np

import matplotlib.pyplot as plt
from scipy import stats
import random

flat = lambda x: x.reshape(-1, 196)
l2_point_dist = lambda x, y: np.linalg.norm(x-y)
l2_98_dist = lambda a, b: sum([l2_point_dist(a[i:i+2], b[i:i+2])/(l2_point_dist(b[120:122], b[144:146])) for i in range(int(len(a)/2))])/98.0
# l2_98_dist = lambda a, b: sum([l2_point_dist(a[i:i+2], b[i:i+2])/(l2_point_dist(b[120:122], b[144:146])) for i in range(33, 98)])/66.0

def calc_dist(filename):
    test_landmarks_gt, test_landmarks_fake, test_landmarks_real = torch.load(filename, map_location=lambda storage, loc: storage)
    test_landmarks_gt, test_landmarks_fake, test_landmarks_real = flat(test_landmarks_gt), flat(test_landmarks_fake), flat(test_landmarks_real)
    dist_1 = [l2_98_dist(test_landmarks_fake[i], test_landmarks_gt[i]) for i in range(len(test_landmarks_gt))]
    dist_2 = [l2_98_dist(test_landmarks_real[i], test_landmarks_gt[i]) for i in range(len(test_landmarks_gt))]
    
    return dist_1, dist_2

def get_couple_dist(filename):
    fake_lands, real_lands, truth_lands = torch.load(filename, map_location=lambda storage, loc: storage)
    dist_1 = [l2_98_dist(np.array(fake_lands[i]), np.array(truth_lands[i])) for i in range(len(truth_lands))]
    dist_2 = [l2_98_dist(np.array(real_lands[i]), np.array(truth_lands[i])) for i in range(len(truth_lands))]
    return dist_1, dist_2

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def get_couple_xy(dist):
    x1 = np.sort(dist)
    y1 = np.arange(len(x1))/float(len(x1))
    return x1, y1

def get_eval_result(filename, figname='../eval/temp.jpg', xlim=0.4, epoch=0):
    dist_29, dist_30 = get_couple_dist(filename)

    x29, y29 = get_couple_xy(dist_29)
    x30, y30 = get_couple_xy(dist_30)
    
    fake_loss, real_loss = (x29.sum()/len(x29))*100, (x30.sum()/len(x30))*100
    
    #clean version
    plt.figure(figsize=(8, 8))
    plt.xlabel('Normalized error')
    plt.ylabel('Proportion of detected landmarks')
    
    c = randomcolor()
    plt.plot(x29, y29, label='eval_fake', color=c, linestyle='dashed')
    plt.plot(x30, y30, label='eval_real', color=c)
    
    plt.xlim([0, xlim])
    plt.ylim([0, 1.0])
    
    plt.title('epoch: %04d nme: fake %.4f | real: %.4f' % (epoch, (x29.sum()/len(x29))*100, (x30.sum()/len(x29))*100))
    plt.legend(bbox_to_anchor=(0.95, 0.15), loc=1, borderaxespad=0.)
    plt.savefig(figname)
    
    return fake_loss, real_loss


