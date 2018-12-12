import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='LAB setup')
    parser.add_argument('--data-dir', default='/home/ubuntu/FaceDatasets/WFLW/WFLW_crop_annotations/resplit_0.0_1_sigma_3_64_train.txt', help='')
    parser.add_argument('--img-channels', type=int, default=3, help='The number of input image channels')
    parser.add_argument('--tensorboard-path', type=str, default='/home/ubuntu/TB/TB24', help='')
    parser.add_argument('--save-params-path', type=str, default='/home/ubuntu/param,24')
    parser.add_argument('--hourglass-channels', type=int, default=256, help='The number of hourglass image channels')
    parser.add_argument('--landmarks', type=int, default=98, help='The number of landmarks')
    parser.add_argument('--boundary', type=int, default=13, help='The number of boundaries')
    parser.add_argument('--per-batch', type=int, default=2, help='Per card batch size')
    parser.add_argument('--epochs', type=int, default=300, help="The number of epochs")
    parser.add_argument('--lr-epoch', type=int, default=200, help="consine end epoch")
    parser.add_argument('--grad-clip', type=float, default=0.1, help='Avoid Gradient Explosion')
    parser.add_argument('--wd',type=float, default=0, help='Weight Decay')

    parser.add_argument('--coe', type=tuple, default=(1, 1/30000, 1), help="Model loss coefficient")
    # 0.1 ok
    parser.add_argument('--lr-base', type=tuple, default=(1e-9, 1e-4, 1e-3), help='Learning Rate')
    parser.add_argument('--lr-target', type=tuple, default=(1e-13, 1e-13, 1e-10), help='Learning Rate')
    parser.add_argument('--lr-mode', type=str, default="cycleCosine", help='Learning Rate')
    parser.add_argument('--beta', type=tuple, default=(0.5, 0.999), help='Adam beta tuple')
    parser.add_argument('--mixup-epoch', type=int, default=0, help='Whether to use miuup')
    parser.add_argument('--mixup-alpha', type=float, default=0.2, help='')

    parser.add_argument('--norm-type', type=str, default='GN', help=' ')
    parser.add_argument('--num-group', type=int, default=8, help='The number of group norm')

    parser.add_argument('--mpl', type=int, default=0, help='Whether to use MPL')
    parser.add_argument('--pretrained', type=int, default=0, help='')
    parser.add_argument('--loss-type', type=int, default=0, help='')
    parser.add_argument('--use-compress', type=int, default=1, help='heatmap compress')
    parser.add_argument('--lr-period', type=int, default=4, help='LR period')


    args = parser.parse_args()
    return args
