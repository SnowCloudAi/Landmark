# -*- coding:utf-8 -*-
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'networks'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'datasets'))
import torch
from datasets.wflw import WFLW
from datasets.newWflw import newWflw
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
from lab_utils import lrSchedule, figPicture, figPicture2Land, figPicture2Land2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# Ignore warnings
import warnings


def heatmapLoss(pred, target):
    target = target.cpu()
    # [B, 13, 64, 64] x 4
    mse = lambda x, y: F.mse_loss(x.cpu(), y, reduction='elementwise_mean')
    loss = 0
    for i in range(len(pred) - 1):
        loss += mse(pred[i], target)
    loss += mse(pred[-1], target)
    return loss.cuda()


def landmarkLoss(pred, target):
    # B, 196
    loss = F.mse_loss(pred.cpu(), target.cpu(), reduction='elementwise_mean')
    return loss.cuda()


def save_checkpoint(path):
    path, time = path.split(',')
    state = {
        'model': models[0].state_dict(),
        'optimizer': optimizers[0].state_dict()
    }
    torch.save(state, path + f'/boundaries{time}.pt')

    state = {
        'model': models[1].state_dict(),
        'optimizer': optimizers[1].state_dict()
    }
    torch.save(state, path + f'/landmarks{time}.pt')


def adjustLr(iter, iters, per_epoch, epoch):
    lr = lrSchedule(args.lr_base[1], iter, iters, epoch=epoch, time=args.lr_period, target_lr=args.lr_target[1], 
                     mode='cycleCosine', per_epoch=per_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main():
    per_epoch = len(train_loader)
    final_iters = per_epoch * args.lr_epoch

    for epoch in range(args.epochs):
        model_b.eval()
        model_l.train()
#         for i in range(len(models)):
#             models[i].train()

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
            optimizer.zero_grad()

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
            
            pred_heatmaps = model_b(mixup_data)
            
            pred_landmarks_fake = model_l(data, pred_heatmaps[3].detach())
            loss_landmarks_fake = landmarkLoss(pred_landmarks_fake, landmarks)
            pred_landmarks_real = model_l(data, heatmap)
            loss_landmarks_real = landmarkLoss(pred_landmarks_real, landmarks)
            
            # Calc loss and Get the model grad (range from 0 to 1)
            loss = loss_landmarks_fake
            loss.backward()
            
            # Setup grad_scale
#             for i in range(len(models)):
#                 torch.nn.utils.clip_grad_value_(models[i].parameters(), args.grad_clip)
            
            # Update model
            optimizer.step()

            
            # The 8 cards average output
            average_loss = hvd.allreduce(loss, True, name='loss_landmarks')
        
            # Others
            if hvd.rank() is 0:
                pbar.set_description(f'Epoch %03d | Loss: %.4f' % (epoch, average_loss.item()))
                writer.add_scalars('Net/loss',
                                   {'landmarks': average_loss.item(), }, global_iters)
                
                writer.add_scalars('LR',
                                   {'regression': optimizer.param_groups[0]['lr'], }, global_iters)
                # if global_iters % 10 == 0:
                #     writer.add_image('Image')
                pic = figPicture(mixup_data[0].cpu().detach().numpy(), mixup_heatmap[0].cpu().detach().numpy(),
                                 pred_heatmaps[3][0].cpu().detach().numpy())
                plt.imsave(f'/home/ubuntu/pic/par5/debug.png', pic)
                
                pic = figPicture2Land2(mixup_data[0].cpu().detach().numpy(), mixup_heatmap[0].cpu().detach().numpy(),
                                 pred_heatmaps[3][0].cpu().detach().numpy(), pred_landmarks_fake[0].cpu().detach().numpy().astype(int), 
                                 pred_landmarks_real[0].cpu().detach().numpy().astype(int), landmarks[0].cpu().detach().numpy().astype(int))
                plt.imsave(f'/home/ubuntu/pic/par5/debug_lands.png', pic)
                
        if hvd.rank() is 0:
            pbar.close()
            
            if (epoch+1) % 50 == 0 and not torch.isnan(
                    loss_landmarks_fake):
                path, time = args.save_params_path.split(',')
                # save_checkpoint(args.save_params_path, epoch)
                torch.save(model_l, path + f"/{time}-model-landmark-{epoch+1}.pkl")
                print('Saved...')
            if epoch % 4 is 3:
                pic = figPicture(mixup_data[0].cpu().detach().numpy(), mixup_heatmap[0].cpu().detach().numpy(),
                                 pred_heatmaps[3][0].cpu().detach().numpy())
                plt.imsave(f'/home/ubuntu/pic/par5/bounds_{epoch}.png', pic)
                pic = figPicture2Land2(mixup_data[0].cpu().detach().numpy(), mixup_heatmap[0].cpu().detach().numpy(),
                                 pred_heatmaps[3][0].cpu().detach().numpy(), pred_landmarks_fake[0].cpu().detach().numpy().astype(int), 
                                 pred_landmarks_real[0].cpu().detach().numpy().astype(int), landmarks[0].cpu().detach().numpy().astype(int))
                plt.imsave(f'/home/ubuntu/pic/par5/lands_{epoch}.png', pic)
    
    if hvd.rank() is 0:
        # Verification per epoch
        writer.close()


if __name__ == '__main__':
    # Init horovod and torch.cuda
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
    train_dataset = newWflw(mode='train')
    train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.per_batch, sampler=train_sampler)

    # Model
    model_b = FAN(4).cuda()
    model_l = LandmarksRegressor(channels=args.hourglass_channels).cuda()
    
    # Load pretrained Model
    if hvd.rank() == 0:
        model_b = torch.load('/home/ubuntu/param/2-model-200.pkl').cuda()
        model_l = torch.load('/home/ubuntu/param/5-model-landmark-200.pkl').cuda()
    hvd.broadcast_parameters(model_b.state_dict(), root_rank=0)
    hvd.broadcast_parameters(model_l.state_dict(), root_rank=0)

    # Optimizer
    optimizer = optim.Adam(model_l.parameters(), lr=args.lr_base[1], betas=args.beta, weight_decay=args.wd)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model_l.named_parameters())

    # Main function
    if hvd.rank() is 0:
        print('Training......')
    main()
