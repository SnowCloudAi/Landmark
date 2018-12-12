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
    DiscriL2, LandmarksRegressor
from networks.newNet import FAN
from lab_args import parse_args
from shutil import rmtree
import horovod.torch as hvd
from torch.backends import cudnn
from torch.utils.data.distributed import DistributedSampler
from lab_utils import lrSchedule, figPicture
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# Ignore warnings
import warnings


def heatmapLoss(pred, target, loss_type):
    loss = 0
    target = target.cpu()
    if loss_type == 0:
        # [B, 13, 64, 64] x 4
        for i in range(len(pred) - 1):
            loss += F.mse_loss(pred[i].cpu(), target)
        loss += 2 * F.mse_loss(pred[-1].cpu(), target)
    elif loss_type == 1:
        for i in range(len(pred)):
            loss += model_dirsc(target, pred[i])
    return loss / (len(pred) + 1)


def landmarkLoss(pred, target):
    # B, 196
    loss = F.mse_loss(pred, target, reduction='elementwise_mean')
    return loss


#
# def save_checkpoint(path, epoch):
#     path, time = path.split(',')
#     state = {
#         'model': model.state_dict(),
#         'optimizer': optimizer.state_dict()
#     }
#     # torch.save(net, path + 'net.pkl')
#     torch.save(state, path + f'/boundaries{time}.pt')
#
#     state = {
#         'model': model_dirsc.state_dict(),
#         'optimizer': optim_dirsc.state_dict()
#     }
#     # torch.save(net, 'net.pkl')
#     torch.save(state, path + f'/dirscLoss{time}.pt')


def adjustLr(iter, iters, per_epoch, epoch):
    lrs = [lrSchedule(args.lr_base[0], iter, iters, epoch=epoch, time=args.lr_period, target_lr=args.lr_target[0],
                      mode=args.lr_mode, per_epoch=per_epoch), ]
    lrs.append(lrSchedule(args.lr_base[1], iter, iters, epoch=epoch, time=args.lr_period, target_lr=args.lr_target[1],
                          mode=args.lr_mode, per_epoch=per_epoch))
    for i in range(len(optimizers)):
        for param_group in optimizers[i].param_groups:
            param_group['lr'] = lrs[i]
    return lrs


def main():
    per_epoch = len(train_loader)
    final_iters = per_epoch * args.lr_epoch
    for epoch in range(args.epochs):
        for i in range(len(models)):
            models[i].train()

        if hvd.rank() == 0:
            pbar = tqdm(train_loader)
        else:
            pbar = train_loader
        for iter_idx, (data, heatmap, landmarks) in enumerate(pbar):
            # Setup
            global_iters = epoch * per_epoch + iter_idx
            lrs = adjustLr(global_iters, final_iters, per_epoch, epoch)
            lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)

            # Clean model grad, otherwise append
            for i in range(len(optimizers)):
                optimizers[i].zero_grad()
                # optimizers[i]
            if args.loss_type == 1:
                optim_dirsc.zero_grad()

            # Put the data to GPUs
            data, heatmap= data.cuda(), heatmap.cuda()

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
            pred_heatmap = models[0](mixup_data)
            loss_heatmap = heatmapLoss(pred_heatmap, mixup_heatmap, args.loss_type)

            pred_real_landmarks = models[1](mixup_data, mixup_heatmap)
            loss_real_landmarks = F.mse_loss(pred_real_landmarks.cpu(), landmarks, reduction='elementwise_mean')
            pred_fake_landmarks = models[1](mixup_data, pred_heatmap[3])
            loss_fake_landmarks = F.mse_loss(pred_fake_landmarks.cpu(), landmarks, reduction='elementwise_mean')

            loss1 = args.coe[0] * loss_heatmap
            loss2 = args.coe[1] * (loss_real_landmarks + loss_fake_landmarks) * 0.5
            loss = loss1 + loss2
            # loss_heatmap = model_dirsc(mixup_heatmap, pred_heatmap)
            # Calc loss and Get the model grad (range from 0 to 1)
            # for i in range(len(optimizers)):
            #     optimizers[i]._register_hooks()
            loss.backward()

            # Setup grad_scale
            # torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)

            # Update model
            for i in range(len(optimizers)):
                optimizers[i].step()
            if args.loss_type == 1:
                optim_dirsc.step()

            # The n cards average output
            loss_tb = [hvd.allreduce(loss1, True, name='loss_heatmap'),
                       hvd.allreduce(loss2, True, name='loss_landmarks'),
                       hvd.allreduce(loss, True, name='loss_sum')]

            # Others
            if hvd.rank() == 0:
                numpyFormat = lambda x: x.cpu().detach().numpy()
                pic = figPicture(numpyFormat(mixup_data[0]), numpyFormat(mixup_heatmap[0]),
                                 numpyFormat(pred_heatmap[3][0]), numpyFormat(pred_real_landmarks[0]),
                                 numpyFormat(pred_fake_landmarks[0]))
                time = args.save_params_path.split(',')[1]
                plt.imsave(f'/home/ubuntu/pic/par{time}/debug.png', pic)
                pbar.set_description(f'Epoch {epoch}  Loss: {loss_tb[2].item()}')
                writer.add_scalars('Net/loss', {'heatmap': loss_tb[0].item(),
                                                'landmarks': loss_tb[1].item(),
                                                'ALL': loss_tb[2].item()}, global_iters)
                writer.add_scalars('LR/lr', {'heatmap': lrs[0],
                                             'landmark': lrs[1]}, global_iters)

        if hvd.rank() == 0:
            pbar.close()
            if epoch % 4 is 3:
                pic = figPicture(data[0].cpu().detach().numpy(), heatmap[0].cpu().detach().numpy(),
                                 pred_heatmap[3][0].cpu().detach().numpy())
                time = args.save_params_path.split(',')[1]
                plt.imsave(f'/home/ubuntu/pic/par{time}/{epoch}.png', pic)
            if epoch % (args.epochs // args.lr_period) == args.epochs // args.lr_period - 1 and not torch.isnan(
                    loss_heatmap):
                path, time = args.save_params_path.split(',')
                # save_checkpoint(args.save_params_path, epoch)
                torch.save(models[0], path + f"/{time}-boundary-{epoch+1}.pkl")
                torch.save(models[1], path + f"/{time}-landmark-{epoch+1}.pkl")
                # if args.loss_type == 1:
                #     torch.save(model_dirsc, path + f"/model_dirsc-{time}.pkl")
                print('Saved...')

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
    train_dataset = newWflw(mode='train')
    train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.per_batch, sampler=train_sampler)

    # Model
    Estimator = BoundaryHeatmapEstimatorwithMPL if args.mpl else BoundaryHeatmapEstimator
    models = [FAN(4).cuda(), LandmarksRegressor(256, 13 + 3, 'GN', 'prelu', 8).cuda()]

    # Load pretrained models
    if hvd.rank() == 0:
        models[0] = torch.load('/home/ubuntu/param/1-model-200.pkl').cuda()
    hvd.broadcast_parameters(models[0].state_dict(), root_rank=0)
    # Optimizer
    # optimizer = optim.SGD(models.parameters(), lr=args.lr_base[0], momentum=0.9, weight_decay=args.wd)
    optimizers = [optim.Adam(models[0].parameters(), lr=args.lr_base[0], betas=args.beta, weight_decay=args.wd)]
    optimizers.append(optim.SGD(models[1].parameters(), lr=args.lr_base[1], momentum=0.9, weight_decay=args.wd))
    optimizers = [hvd.DistributedOptimizer(optimizers[i], named_parameters=models[i].named_parameters()) for i in
                 range(2)]

    if args.loss_type == 1:
        models_dirsc = DiscriL2(args.boundary).cuda()
        optim_dirsc = optim.SGD(models_dirsc.parameters(), lr=args.lr_target[2], momentum=0.9, weight_decay=args.wd)
        optim_dirsc = hvd.DistributedOptimizer(optim_dirsc, models_dirsc.named_parameters())

    # Main function
    if hvd.rank() == 0:
        print('Training......')
    main()
