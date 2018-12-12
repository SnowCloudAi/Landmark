_# -*- coding:utf-8 -*-
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
    DiscriL2
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
    if loss_type == 0:
        # [B, 13, 64, 64] x 4
        for i in range(len(pred) - 1):
            loss += F.mse_loss(pred[i], target)
        loss += 2 * F.mse_loss(pred[-1], target)
    elif loss_type == 1:
        for i in range(len(pred)):
            loss += model_dirsc(target, pred[i])
    return loss / (len(pred) + 1)


def save_checkpoint(path, epoch):
    path, time = path.split(',')
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    # torch.save(net, path + 'net.pkl')
    torch.save(state, path + f'/boundaries{time}.pt')

    state = {
        'model': model_dirsc.state_dict(),
        'optimizer': optim_dirsc.state_dict()
    }
    # torch.save(net, 'net.pkl')
    torch.save(state, path + f'/dirscLoss{time}.pt')


def adjustLr(iter, iters, per_epoch, epoch):
    lr = lrSchedule(args.lr_base[0], iter, iters, epoch=epoch, time=args.lr_period, target_lr=args.lr_target[0],
                    mode='cycleCosine', per_epoch=per_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main():
    per_epoch = len(train_loader)
    final_iters = per_epoch * args.lr_epoch
    for epoch in range(args.epochs):
        model.train()

        if hvd.rank() == 0:
            pbar = tqdm(train_loader)
        else:
            pbar = train_loader
        for iter_idx, (data, heatmap, landmarks) in enumerate(pbar):
            # Setup
            global_iters = epoch * per_epoch + iter_idx
            if epoch < args.lr_epoch:
                lr = adjustLr(global_iters, final_iters, per_epoch, epoch)
            else:
                lr = args.lr_target[0]
            lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)

            # Clean model grad, otherwise append
            optimizer.zero_grad()
            if args.loss_type == 1:
                optim_dirsc.zero_grad()

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
            # loss_heatmap = model_dirsc(mixup_heatmap, pred_heatmap)
            # Calc loss and Get the model grad (range from 0 to 1)
            loss_heatmap.backward()

            # Setup grad_scale
            # torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)

            # Update model
            optimizer.step()
            if args.loss_type == 1:
                optim_dirsc.step()

            # The 8 cards average output
            loss_tb = hvd.allreduce(loss_heatmap, True, name='loss_heatmap')

            # Others
            if hvd.rank() == 0:
                pic = figPicture(mixup_data[0].cpu().detach().numpy(), mixup_heatmap[0].cpu().detach().numpy(),
                                 pred_heatmap[3][0].cpu().detach().numpy())
                plt.imsave(f'/home/ubuntu/pic/par1/debug.png', pic)
                pbar.set_description(f'Epoch {epoch}  Loss: {loss_tb}')
                writer.add_scalar('Net/loss_heatmap', loss_tb, global_iters)
                writer.add_scalar('LR/lr', lr, global_iters)

        if hvd.rank() == 0:
            pbar.close()
            if epoch % 4 is 3:
                pic = figPicture(data[0].cpu().detach().numpy(), heatmap[0].cpu().detach().numpy(),
                                 pred_heatmap[3][0].cpu().detach().numpy())
                plt.imsave(f'/home/ubuntu/pic/par1/{epoch}.png', pic)
            if epoch % (args.epochs // args.lr_period) == args.epochs // args.lr_period - 1 and not torch.isnan(
                    loss_heatmap):
                path, time = args.save_params_path.split(',')
                # save_checkpoint(args.save_params_path, epoch)
                torch.save(model, path + f"/{time}-model-{epoch+1}.pkl")
                if args.loss_type == 1:
                    torch.save(model_dirsc, path + f"/model_dirsc-{time}.pkl")
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
    model = FAN(4).cuda()

    # Load pretrained Model
    if hvd.rank() == 0:
        model = torch.load('/home/ubuntu/param/0-model-200.pkl').cuda()
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    # Optimizer
    # optimizer = optim.SGD(model.parameters(), lr=args.lr_base[0], momentum=0.9, weight_decay=args.wd)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_base[0], betas=args.beta, weight_decay=args.wd)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    if args.loss_type == 1:
        model_dirsc = DiscriL2(args.boundary).cuda()
        optim_dirsc = optim.SGD(model_dirsc.parameters(), lr=args.lr_target[2], momentum=0.9, weight_decay=args.wd)
        optim_dirsc = hvd.DistributedOptimizer(optim_dirsc, model_dirsc.named_parameters())



    # Main function
    if hvd.rank() == 0:
        print('Training......')
    main()