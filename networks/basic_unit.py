from torch import nn

#
# def conv_unit(in_channels, out_channels, kernel_size, stride, use_bn=False, act_type=None):
#     pad = kernel_size // 2
#     layers = []
#     layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
#     if use_bn:
#         layers.append(nn.BatchNorm2d(out_channels))
#     if act_type is 'prelu':
#         layers.append(nn.PReLU())
#     elif act_type is 'relu':
#         layers.append(nn.ReLU())
#     return layers
#
#
# def gan_conv_unit(in_channels, out_channels, kernel_size, stride, use_in=False, use_leaky=False, slope=0.2):
#     pad = kernel_size // 2
#     layers = []
#     layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
#     if use_in:
#         layers.append(nn.InstanceNorm2d(out_channels))
#     if use_leaky:
#         layers.append(nn.LeakyReLU(slope))
#     return layers


def optUnit(opt_type=None, norm_type=None, act_type=None, in_ch=0, out_ch=0, ker_size=0, stride=0, bias=False,
            conv_groups=1,
            num_group=None, affine=True,
            slope=0.2):
    pad = ker_size // 2
    layers = []

    # Opt
    if opt_type is 'conv':
        layers.append(nn.Conv2d(in_ch, out_ch, ker_size, stride, pad, bias=bias, groups=conv_groups))
    elif opt_type is "linear":
        layers.append(nn.Linear(in_ch, out_ch, bias))

    # Norm
    if norm_type is 'BN':
        layers.append(nn.BatchNorm2d(out_ch, affine=affine))
    elif norm_type is 'IN':
        layers.append(nn.InstanceNorm2d(out_ch, affine=affine))
    elif norm_type is 'GN':
        layers.append(nn.GroupNorm(num_group, out_ch, affine=affine))

    # Act
    if act_type is 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif act_type is 'prelu':
        layers.append(nn.PReLU())
    elif act_type is 'leakyRelu':
        layers.append(nn.LeakyReLU(slope, inplace=True))

    return layers