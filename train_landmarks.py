# -*- coding:utf-8 -*-
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'networks'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'datasets'))
import torch
from datasets.wflw import WFLW
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
from networks.net import BoundaryHeatmapEstimator, LandmarksRegressor, Discriminator
from lab_args import parse_args
from shutil import rmtree
import horovod.torch as hvd
from torch.backends import cudnn
from torch.utils.data.distributed import DistributedSampler
from lab_utils import lrSchedule
import numpy as np


def heatmapLoss(pred, target):
    target = target.cpu()
    # [B, 13, 64, 64] x 4
    mse = lambda x, y: F.mse_loss(x.cpu(), y, reduction='elementwise_mean')
    loss = 0
    for i in range(len(pred) - 1):
        loss += mse(pred[i], target)
    loss += mse(pred[-1], target)
    return loss


def landmarkLoss(pred, target):
    # B, 196
    loss = F.mse_loss(pred.cpu(), target.cpu(), reduction='elementwise_mean')
    return loss


def wganLoss(real, fake):
    loss = F.l1_loss(real.cpu(), fake.cpu(), reduction='elementwise_mean')
    return loss


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

    state = {
        'model': models[2].state_dict(),
        'optimizer': optimizers[2].state_dict()
    }
    torch.save(state, path + f'/gan{time}.pt')


def adjustLr(iter, iters):
    lr = [lrSchedule(args.lr_base[i], iter, iters, target_lr=args.lr_target[i], mode=args.lr_mode) for i in range(3)]
    for i in range(3):
        for param_group in optimizers[i].param_groups:
            param_group['lr'] = lr[i]


def main():
    per_epoch = len(train_loader)
    final_iters = per_epoch * args.lr_epoch

    for epoch in range(args.epochs):

        for i in range(len(models)):
            models[i].train()

        if hvd.rank() is 0:
            pbar = tqdm(train_loader)
        else:
            pbar = train_loader
        for iter_idx, (data, heatmap, landmarks) in enumerate(pbar):
            # Setup
            global_iters = epoch * per_epoch + iter_idx
            if epoch > args.lr_epoch:
                adjustLr(global_iters, final_iters)
            lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)

            # Clean model grad, otherwise append
            for i in range(len(optimizers)):
                optimizers[i].zero_grad()

            # Put the data to GPUs
            data, heatmap, landmarks = data.cuda(), (heatmap * np.sqrt(2 * np.pi * 4)).cuda(), landmarks.cuda()

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

            pred_landmarks_fake = models[1](data, pred_heatmaps[3])
            loss_landmarks_fake = landmarkLoss(pred_landmarks_fake, landmarks)
            pred_landmarks_real = models[1](data, heatmap)
            loss_landmarks_real = landmarkLoss(pred_landmarks_real, landmarks)

            real = models[2](data, heatmap, True)
            fake = models[2](data, pred_heatmaps[3], False)
            loss_gan = wganLoss(real, fake)

            # Calc loss and Get the model grad (range from 0 to 1)
            loss1 = args.coe[0] * loss_heatmap
            loss2 = args.coe[1] * (loss_landmarks_fake + loss_landmarks_real)
            loss3 = args.coe[2] * loss_gan
            loss = loss1 + loss2 + loss3
            loss.backward()

            # Setup grad_scale
            models[0].heatmap[0].weight.grad.data *= 0.25
            for i in range(len(models)):
                torch.nn.utils.clip_grad_value_(models[i].parameters(), args.grad_clip)

            # Update model
            for i in range(len(optimizers)):
                optimizers[i].step()

            # The 8 cards average output
            average_loss = [hvd.allreduce(loss1, True, name='loss_heatmap'),
                            hvd.allreduce(loss2, True, name='loss_landmark'),
                            hvd.allreduce(loss3, True, name='loss_gan'), hvd.allreduce(loss, True, name='loss_sum')]
            # Others
            if hvd.rank() is 0:
                pbar.set_description(f'Epoch {epoch}  ')
                writer.add_scalars('Net/loss',
                                   {'heatmap': average_loss[0].item(),
                                    'landmarks': average_loss[1].item(),
                                    'GAN': average_loss[2].item(),
                                    'ALL': average_loss[3].item()}, global_iters)

                writer.add_scalars('LR',
                                   {'estimation': optimizers[0].param_groups[0]['lr'],
                                    'regression': optimizers[1].param_groups[0]['lr'],
                                    'gan': optimizers[2].param_groups[0]['lr'],
                                    }, global_iters)
                # if global_iters % 10 == 0:
                #     writer.add_image('Image')

        if hvd.rank() is 0:
            pbar.close()
            if epoch % 10 == 0 and epoch > 200 and not torch.isnan(average_loss[3]):
                save_checkpoint(args.save_params_path)
                print('Saved...')

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
    train_dataset = WFLW('train', path=args.data_dir)
    train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.per_batch, sampler=train_sampler)

    # Model
    # norm实现有问题
    models = [BoundaryHeatmapEstimator(args.img_channels, args.hourglass_channels, args.boundary, ).cuda(), ]
    models.append(LandmarksRegressor(channels=args.hourglass_channels).cuda())
    models.append(Discriminator(args.boundary).cuda())

    # Optimizer
    optimizers = [optim.Adam(models[i].parameters(), lr=args.lr_base[i], betas=args.beta, weight_decay=args.wd) for i in
                  range(3)]
    optimizers = [hvd.DistributedOptimizer(optimizers[i], named_parameters=models[i].named_parameters()) for i in
                  range(3)]

    # Main function
    if hvd.rank() is 0:
        print('Training......')
    main()
