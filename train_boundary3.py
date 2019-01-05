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
import tensorflow as tf
from tensorboardX import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
from networks.net import BoundaryHeatmapEstimator, BoundaryHeatmapEstimatorwithMPL, \
    DiscriL2
from networks.newNet import FAN
from lab_args import parse_args
from shutil import rmtree
import horovod.torch as hvd
from torch.backends import cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lab_utils import lrSchedule, figPicture
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# Ignore warnings
import warnings
import random
import cv2

def heatmapLoss(pred, target, loss_type):
    loss = 0
    if loss_type == 0:
        # [B, 13, 64, 64] x 4
        for i in range(len(pred) - 1):
            loss += F.mse_loss(pred[i], target)
        loss += 2 * F.mse_loss(pred[-1], target)
    elif loss_type == 1:
        for i in range(len(pred)):
            loss += model_dirsc(target, pred[i])
    return loss / (len(pred) + 1)

def adjustLr(iter, iters, per_epoch, epoch):
    lr = lrSchedule(args.lr_base[0], iter, iters, epoch=epoch, time=args.lr_period, target_lr=args.lr_target[0],
                    mode='cycleCosine', per_epoch=per_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def evaluate(model):
    loss_list = []
    model.eval()
    
    with tqdm(test_loader) as pbar:
        for iter_idx, (data, heatmap, landmarks) in enumerate(pbar):
            with torch.no_grad():
                data, heatmap = data.cuda(), heatmap.cuda()
                pred_heatmaps = model(data)
                loss_heatmaps = heatmapLoss(pred_heatmaps, heatmap, loss_type=0)
            loss_list.append(loss_heatmaps.item())
#         if iter_idx == 0:
#             break
    return sum(loss_list)/len(test_loader)

def main():
    per_epoch = len(train_loader)
    final_iters = per_epoch * args.lr_epoch
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train_loader.dataset.annotations)

        pbar = tqdm(train_loader)
        
        for iter_idx, (data, heatmap, landmarks) in enumerate(pbar):
            # Setup
            global_iters = epoch * per_epoch + iter_idx
            
            #if epoch < args.lr_epoch:
            #    lr = adjustLr(global_iters, final_iters, per_epoch, epoch)
            #else:
            #    lr = args.lr_target[0]
            lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)

            # Clean model grad, otherwise append
            optimizer.zero_grad()

            # Put the data to GPUs
            data, heatmap = data.cuda(), heatmap.cuda()

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
            pred_heatmap = model(mixup_data)
            loss_heatmap = heatmapLoss(pred_heatmap, mixup_heatmap, args.loss_type)
            
            # Calc loss and Get the model grad (range from 0 to 1)
            loss_heatmap.backward()

            # Setup grad_scale
            torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)

            # Update model
            optimizer.step()

            # The 8 cards average output
            loss_tb = hvd.allreduce(loss_heatmap, True, name='loss_heatmap')
            
            # Others
            if hvd.rank() == 0:
                x = pred_heatmap[3]
                x = (x-x.min())/(x.max()-x.min())
                x = (x > 0.20).float() * x
                pic = figPicture(mixup_data[0].cpu().detach().numpy(), mixup_heatmap[0].cpu().detach().numpy(),
                                 x[0].cpu().detach().numpy())
                pic = cv2.resize(pic, (896, 512))
                text = 'Epoch: %s Loss: %.6f LR: %.7f' % (epoch+1, loss_heatmap.item(), optimizer.param_groups[0]['lr'])
                cv2.putText(pic, text, (128, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
                plt.imsave('../pic/par3/debug.png', pic)

                pbar.set_description(text)
                writer.add_scalar('Net/loss_heatmap', loss_heatmap, global_iters)
                writer.add_scalar('LR/lr', optimizer.param_groups[0]['lr'], global_iters)
        
        if hvd.rank() == 0:
            pbar.close()

            # evaluate and adjust lr chance
            if (epoch+1) % 1 is 0:
                print('\nevaluating...')
                eval_loss = evaluate(model)
                print('\nevaluating loss: %.6f' % eval_loss)
                scheduler.step(eval_loss)
                writer.add_scalar('Net/loss_heatmap_eval', eval_loss, global_iters)
            
            if (epoch+1) % 1 is 0:
                pic = figPicture(data[0].cpu().detach().numpy(), heatmap[0].cpu().detach().numpy(),
                                 pred_heatmap[3][0].cpu().detach().numpy())
                pic = cv2.resize(pic, (896, 512))
                text = 'Epoch: %s Loss: %.6f LR: %.7f' % (epoch+1, loss_heatmap.item(), optimizer.param_groups[0]['lr'])
                pic = cv2.putText(pic, text, (128, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
                plt.imsave('../pic/par3/%s.png' % (epoch), pic)
            
            if (epoch+1) % 1 is 0 and not torch.isnan(loss_heatmap):
                path, time = args.save_params_path.split(',')
                # save_checkpoint(args.save_params_path, epoch)
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(model, path + "/%s-model-boundary-%s.pkl" % (time, epoch+1))
                
                print('\nSaved at "%s/%s-model-boundary-%s.pkl"' % (path, time, epoch+1))
                
                x = pred_heatmap[3]
                x = (x-x.min())/(x.max()-x.min())
                x = (x > 0.20).float() * x
                pic = figPicture(mixup_data[0].cpu().detach().numpy(), mixup_heatmap[0].cpu().detach().numpy(),
                                 x[0].cpu().detach().numpy())
                text = "%s-model-boundary-%s.pkl" % (time, epoch+1)
                pic = np.moveaxis(pic, 1, 0)
                pic = np.moveaxis(pic, -1, 0)
                writer.add_image(text, pic, global_iters)

    if hvd.rank() == 0:
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
    
    if hvd.rank() == 0:
        # Announce
        print(args)

        # Init tensorboard
        rmtree(args.tensorboard_path, ignore_errors=True)
        writer = SummaryWriter(args.tensorboard_path)

    # DataLoader
    train_dataset = iWflw(mode='train', transform_mode='train')
    train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.per_batch, sampler=train_sampler)
    
    test_dataset = iWflw(mode='test', transform_mode='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.per_batch)

    # Model
    model = FAN(4).cuda()

    # Load pretrained Model
    if hvd.rank() == 0:
        model = torch.load('../param/exp2/0-model-boundary-18.pkl', map_location=lambda storage, loc: storage).cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr_base[0], betas=args.beta, weight_decay=args.wd)
    # forgot to add this line before
    # optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True, threshold=5e-5, threshold_mode='rel', cooldown=0, min_lr=1e-8)
    
    # Main function
    if hvd.rank() == 0:
        
        print('\nfirst evaluating...')
        eval_loss = evaluate(model)
        print('\nfirst evaluating loss: %.6f' % eval_loss)
        scheduler.step(eval_loss)
        
        print('\nstart training......')
    main()

