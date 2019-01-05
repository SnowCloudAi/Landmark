# -*- coding:utf-8 -*-
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'networks'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'datasets'))
import torch
from datasets.wflw import WFLW
from datasets.newWflw import newWflw
from datasets.iWflw import iWflw
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
from networks.net import BoundaryHeatmapEstimator, BoundaryHeatmapEstimatorwithMPL, \
    DiscriL2, LandmarksRegressor, Discriminator
from networks.newNet import FAN
from lab_args import parse_args
from shutil import rmtree
import horovod.torch as hvd
from torch.backends import cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lab_utils import lrSchedule, figPicture, figPicture2Land, figPicture2Land2
from eval_utils import get_eval_result, l2_98_dist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# Ignore warnings
import warnings
import random
import cv2
import math

def heatmapLoss(pred, target):
    target = target.cpu()
    # [B, 13, 64, 64] x 4
    mse = lambda x, y: F.mse_loss(x.cpu(), y, reduction='elementwise_mean')
    loss = 0
    for i in range(len(pred) - 1):
        loss += mse(pred[i], target)
    loss += mse(pred[-1], target)
    return loss.cuda()

def wingLoss(pred, target, w=10.0, epsilon=2.0):
    dis = torch.abs_(pred - target)
    c = w * (1.0 - math.log(w / epsilon + 1.0))
    loss_mat = torch.where(dis < w, w * torch.log(1.0 + dis / epsilon), dis - c)
    loss = torch.mean(torch.sum(loss_mat, dim=1), dim=0)
    return loss.cuda()

def landmarkLoss(pred, target):
    # B, 196
    loss = F.mse_loss(pred.cpu(), target.cpu(), reduction='elementwise_mean')
    return loss.cuda()

def normLoss(pred, target):
    dist = torch.pow((pred - target), 2)
    norm = torch.norm(((pred - target)[:, 120:122] - (pred - target)[:, 144:146]), 2, 1)
    for i in range(10):
        dist[i] /= norm[i]
    dist_all = dist.sum()/(10*196)
    return dist_all.cuda()

def save_checkpoint(path):
    path, time = path.split(',')
    state = {
        'model': models[0].state_dict(),
        'optimizer': optimizers[0].state_dict()
    }
    torch.save(state, path + '/boundaries%s.pt' % (time))

    state = {
        'model': models[1].state_dict(),
        'optimizer': optimizers[1].state_dict()
    }
    torch.save(state, path + '/landmarks%s.pt' % (time))

def adjustLr(iter, iters, per_epoch, epoch):
    lr = [lrSchedule(args.lr_base[i], iter, iters, target_lr=args.lr_target[i], mode=args.lr_mode) for i in range(2)]
    for i in range(2):
        for param_group in optimizers[i].param_groups:
            param_group['lr'] = lr[i]
    return lr

def evaluate_stage2(train_code=0, train_mod='iwflw', clean_mode='nc', epoch=0, figname='../eval/demo.jpg'):
    model_b.eval()
    model_l.eval()
    
    fake_lands = []
    real_lands = []
    truth_lands = []
    with tqdm(test_loader) as pbar:
        for iter_idx, (data, heatmap, landmarks) in enumerate(pbar):
            data, heatmap = data.cuda(), heatmap.cuda()
            with torch.no_grad():
                pred_heatmaps = model_b(data)
                if clean_mode == 'c':
                    x = pred_heatmaps[3]
                    for i in range(batch_size):
                        x[i] = (x[i]-x[i].min())/(x[i].max()-x[i].min())
                        x[i] = (x[i] > 0.20).float() * x[i]
                    pred_landmarks_fake = model_l(data, x.detach())
                    pred_landmarks_real = model_l(data, heatmap)
                else:
                    pred_landmarks_fake = model_l(data, pred_heatmaps[3].detach())
                    pred_landmarks_real = model_l(data, heatmap)
            
            fake_lands += pred_landmarks_fake.detach().cpu().numpy().tolist()
            real_lands += pred_landmarks_real.detach().cpu().numpy().tolist()
            truth_lands += landmarks.cpu().numpy().tolist()
            
#         if iter_idx == 0:
#             break
    filename = '../eval/model_%s_%s_%s_%s_.pkl' % (train_code, epoch, train_mod, clean_mode)
    torch.save([fake_lands, real_lands, truth_lands], filename)
    fake_loss, real_loss = get_eval_result(filename, figname, xlim=0.4, epoch=epoch)
    return fake_loss, real_loss

def main(clean_mode='nc', train_code=0):
    per_epoch = len(train_loader)
    final_iters = per_epoch * args.lr_epoch

    for epoch in range(args.epochs):
        # random.shuffle(train_loader.dataset.annotations)
        
        for i in range(len(models)):
            models[i].train()
        
        if hvd.rank() is 0:
            pbar = tqdm(train_loader)
        else:
            pbar = train_loader
        for iter_idx, (data, heatmap, landmarks) in enumerate(pbar):
            # Setup
            global_iters = epoch * per_epoch + iter_idx
            if epoch < args.lr_epoch:
                lr = adjustLr(global_iters, final_iters, per_epoch, epoch)
            lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
            
            # Clean model grad, otherwise append
            for i in range(len(optimizers)):
                optimizers[i].zero_grad()
            
            # Put the data to GPUs
            data, heatmap, landmarks = data.cuda(), heatmap.cuda(), landmarks.cuda()
            
            # Model forward
            # mixup data mixup可以当作数据增强遮挡
            if epoch < args.mixup_epoch:
                # Can't use [::-1]
                inv_idx = torch.arange(data.shape[0] - 1, -1, -1).long().cuda()
                mixup_data = lam * data + (1 - lam) * data.index_select(0, inv_idx)
                mixup_heatmap = lam * heatmap + (1 - lam) * heatmap.index_select(0, inv_idx)
            else:
                mixup_data = data
                mixup_heatmap = heatmap
            
            pred_heatmaps = models[0](mixup_data)
            loss_heatmap = heatmapLoss(pred_heatmaps, mixup_heatmap)
            
            if clean_mode == 'c':
                x = pred_heatmaps[3]
                for i in range(batch_size):
                    x[i] = (x[i]-x[i].min())/(x[i].max()-x[i].min())
                    x[i] = (x[i] > 0.20).float() * x[i]
            else:
                x = pred_heatmaps[3]
            
            pred_landmarks_fake = models[1](data, x)
            loss_landmarks_fake = wingLoss(pred_landmarks_fake, landmarks, w=10.0, epsilon=2.0)
            # loss_landmarks_fake = normLoss(pred_landmarks_fake, landmarks)
            
            pred_landmarks_real = models[1](data, heatmap)
            loss_landmarks_real = wingLoss(pred_landmarks_real, landmarks, w=10.0, epsilon=2.0)
            # loss_landmarks_real = normLoss(pred_landmarks_real, landmarks)
            
            # Calc loss and Get the model grad (range from 0 to 1)
            #loss1 = loss_heatmap
            #loss2 = (loss_landmarks_fake + loss_landmarks_real)
            #loss = (loss_landmarks_fake + loss_landmarks_real)
            
            loss1 = args.coe[0] * loss_heatmap
            loss2 = args.coe[1] * (loss_landmarks_fake + loss_landmarks_real)
            loss = loss1 + loss2
            
            loss.backward()
            
            # Setup grad_scale
            # torch.nn.utils.clip_grad_value_(model_l.parameters(), args.grad_clip)
            
            # Update model
            for i in range(len(optimizers)):
                optimizers[i].step()
            
            # The 8 cards average output
            average_loss = hvd.allreduce(loss, True, name='loss_landmarks')
            
            # Others
            if hvd.rank() is 0:
                text = 'Epoch %03d | Loss: %.5f | %.5f | %.5f' % (epoch, loss1.item(), loss2.item(), average_loss.item())
                pbar.set_description(text)
                writer.add_scalars('Net/loss',
                                   {'landmarks sum': average_loss.item(), 
                                    'heatmap loss': loss1.item(), 
                                    'landmarks loss': loss2.item(), }, global_iters)
                
                writer.add_scalars('LR',
                                   {'regression': optimizers[0].param_groups[0]['lr'], 
                                   'landmark': optimizers[1].param_groups[0]['lr'], }, global_iters)
                
                pic1 = figPicture(mixup_data[0].cpu().detach().numpy(), mixup_heatmap[0].cpu().detach().numpy(),
                                 x[0].cpu().detach().numpy())                
                pic2 = figPicture2Land2(mixup_data[0].cpu().detach().numpy(), mixup_heatmap[0].cpu().detach().numpy(),
                                 x[0].cpu().detach().numpy(), pred_landmarks_fake[0].cpu().detach().numpy().astype(int), 
                                 pred_landmarks_real[0].cpu().detach().numpy().astype(int), landmarks[0].cpu().detach().numpy().astype(int))
                pic1 = (pic1 -pic1.min())/(pic1.max()-pic1.min())
                pic2 = (pic2 -pic2.min())/(pic2.max()-pic2.min())
                pic_all_ = np.hstack((pic1, pic2))
                pic_all_ = cv2.putText(pic_all_, text, (128, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (1.0), 2)
                plt.imsave('../pic/par%d/debug.png' % (train_code), pic_all_)
                
        
        if hvd.rank() is 0:
            pbar.close()
            
            # evaluate and adjust lr chance
            if (epoch+1) % 1 is 0:
                print('\nevaluating...')
                figname='../pic/par%d/debug_ced_%04d.jpg' % (train_code, epoch+1)
                eval_loss_fake, eval_loss_real = evaluate_stage2(train_code=train_code, train_mod='iwflw', clean_mode='nc', epoch=epoch+1, figname=figname)
                print('\nevaluating loss: %.6f | %.6f' % (eval_loss_fake, eval_loss_real))
                
                for i in range(2):
                    schedulers[i].step(eval_loss_fake+eval_loss_real)
                
                writer.add_scalars('Net/eval_loss',
                                   {'landmarks nms_fake': eval_loss_fake, 
                                    'landmarks nms_real': eval_loss_real,
                                   }, global_iters)
                text = '%s_%.4f_%.4f' % (epoch+1, eval_loss_fake, eval_loss_real)
                
                pic = plt.imread(figname)
                pic = np.moveaxis(pic, 1, 0)
                pic = np.moveaxis(pic, -1, 0)
                writer.add_image(text, pic, global_iters)
            
            # save temporary result
            if (epoch+1) % 1 is 0:
                pic1 = figPicture(mixup_data[0].cpu().detach().numpy(), mixup_heatmap[0].cpu().detach().numpy(),
                                 x[0].cpu().detach().numpy())
                pic2 = figPicture2Land2(mixup_data[0].cpu().detach().numpy(), mixup_heatmap[0].cpu().detach().numpy(),
                                 x[0].cpu().detach().numpy(), pred_landmarks_fake[0].cpu().detach().numpy().astype(int), 
                                 pred_landmarks_real[0].cpu().detach().numpy().astype(int), landmarks[0].cpu().detach().numpy().astype(int))
                pic1 = (pic1 -pic1.min())/(pic1.max()-pic1.min())
                pic2 = (pic2 -pic2.min())/(pic2.max()-pic2.min())
                pic_all_ = np.hstack((pic1, pic2))
                pic_all_ = cv2.putText(pic_all_, text, (128, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (1.0), 2)
                
                filename = '../pic/par%d/debug_%s.png' % (train_code, epoch+1)
                plt.imsave(filename, pic_all_)
            
            # save model
            if (epoch+1) % 1 is 0 and not torch.isnan(loss_landmarks_fake):
                path, time = args.save_params_path.split(',')
                # save_checkpoint(args.save_params_path, epoch)
                if not os.path.exists(path):
                    os.makedirs(path)
                filename = path + "/%s-model-landmark-%s.pkl" % (time, epoch+1)
                torch.save(model_l, filename)
                
                print('\nSaved at %s' % filename)
                
                filename = path + "/%s-model-boundary-%s.pkl" % (time, epoch+1)
                torch.save(model_b, filename)
                
                print('\nSaved at %s' % filename)
                
    
    if hvd.rank() is 0:
        # Verification per epoch
        
        writer.close()


if __name__ == '__main__':
    # Init horovod and torch.cuda
    warnings.filterwarnings("ignore")
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    # Setup
    args = parse_args()

    # This flag allows you to enable the inbuilt cudnn
    # auto-tuner to find the best algorithm to use for your hardware.
    cudnn.benchmark = True

    if hvd.rank() is 0:
        # Announce
        print(args)

        # Init tensorboard
        rmtree(args.tensorboard_path, ignore_errors=True)
        writer = SummaryWriter(args.tensorboard_path)
        
        
    # DataLoader
#     train_dataset = newWflw(mode='train')
    attr_list = ['pose', 'expression', 'illumination', 'make_up', 'occlusion', 'blur', 'all']
    train_dataset = iWflw(mode='train', transform_mode='soft', attr=attr_list[0], balance='True')
    train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.per_batch, sampler=train_sampler)
    
    test_dataset = iWflw(mode='test', transform_mode='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.per_batch)
    
    # Model
    model_b = FAN(4).cuda()
    model_l = LandmarksRegressor(channels=args.hourglass_channels).cuda()

    # Load pretrained Model
    if hvd.rank() == 0:
        model_b = torch.load('../param/exp10/0-model-boundary-100.pkl').cuda()
        model_l = torch.load('../param/exp10/0-model-landmark-100.pkl').cuda()
    hvd.broadcast_parameters(model_b.state_dict(), root_rank=0)
    hvd.broadcast_parameters(model_l.state_dict(), root_rank=0)
    
    models = [model_b, model_l]

    # Optimizer
    optimizers = [optim.Adam(models[i].parameters(), lr=args.lr_base[i], betas=args.beta, weight_decay=args.wd) for i in
                  range(2)]
    optimizers = [hvd.DistributedOptimizer(optimizers[i], named_parameters=models[i].named_parameters()) for i in
                  range(2)]
    schedulers = [ReduceLROnPlateau(optimizers[i], mode='min', factor=0.5, patience=5, verbose=True, threshold=5e-5, threshold_mode='rel', cooldown=0, min_lr=1e-6) for i in range(2)]

    # Main function
    if hvd.rank() is 0:
        print('Training......')
    main(clean_mode='nc', train_code=args.train_code)
