from torch import nn
import torch.nn.functional as F
from networks.basic_unit import optUnit
import torch


class ResidualBlock(nn.Module):
    """
    1: Face
    2: down
    3: normal
    """

    def __init__(self, in_channels, out_channels, stride, res_type=1, channel_fac=4, norm_type='BN', num_group=None,
                 act_type='relu'):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.res_type = res_type
        if res_type is 1:
            layers = optUnit(norm_type=norm_type, num_group=num_group)
            layers += optUnit(opt_type='conv', norm_type=norm_type, act_type=act_type, in_ch=in_channels,
                              out_ch=out_channels, ker_size=3, stride=1, num_group=num_group)
            layers += optUnit(opt_type='conv', in_ch=in_channels, out_ch=out_channels, ker_size=3,
                              stride=stride, num_group=num_group)

            self.right = nn.Sequential(*layers)

            if self.stride is 2:
                self.left = nn.Sequential(
                    *optUnit(opt_type='conv', in_ch=in_channels, out_ch=out_channels, ker_size=3, stride=2,
                             num_group=num_group))


        elif res_type is 2:

            left_layers = optUnit(opt_type='conv', in_ch=in_channels, out_ch=out_channels, ker_size=1, stride=stride,
                                  num_group=num_group)
            self.left = nn.Sequential(*left_layers)
            # bottleneck
            right_layers = optUnit(opt_type='conv', norm_type=norm_type, act_type=act_type, in_ch=in_channels,
                                   out_ch=out_channels // channel_fac, ker_size=1, stride=1, num_group=num_group)

            right_layers += optUnit(opt_type='conv', norm_type=norm_type, act_type=act_type,
                                    in_ch=out_channels // channel_fac, out_ch=out_channels // channel_fac, ker_size=1,
                                    stride=stride, num_group=num_group)
            right_layers += optUnit(opt_type='conv', in_ch=out_channels // channel_fac, out_ch=out_channels, ker_size=1,
                                    stride=1, num_group=num_group)
            self.right = nn.Sequential(*right_layers)

        elif res_type is 3:
            right_layers = optUnit(norm_type=norm_type, act_type=act_type, out_ch=in_channels, num_group=num_group)
            right_layers += optUnit(opt_type='conv', norm_type=norm_type, act_type=act_type, in_ch=in_channels,
                                    out_ch=out_channels // channel_fac, ker_size=1, stride=1, num_group=num_group)
            right_layers += optUnit(opt_type='conv', norm_type=norm_type, act_type=act_type,
                                    in_ch=out_channels // channel_fac, out_ch=out_channels // channel_fac, ker_size=3,
                                    stride=1, num_group=num_group)
            right_layers += optUnit(opt_type='conv', in_ch=out_channels // channel_fac, out_ch=out_channels, ker_size=1,
                                    stride=1, num_group=num_group)

            self.right = nn.Sequential(*right_layers)

    def forward(self, x):
        if self.res_type is 1:
            if self.stride is 1:
                return self.right(x) + x
            else:
                return self.right(x) + self.left(x)
        elif self.res_type is 2:
            return self.left(x) + self.right(x)
        elif self.res_type is 3:
            self.left = x
            return self.left + self.right(x)


class HourglassBlock(nn.Module):
    """
        Channels and Resolutions Consistent
    """

    def __init__(self, channels, norm_type='BN', act_type='prelu', num_group=None):
        super(HourglassBlock, self).__init__()
        # Lower 64
        self.lower_res0_branch = ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group,
                                               act_type=act_type)
        # pooling1 32
        self.lower_res1_main = nn.Sequential(
            *[nn.MaxPool2d(2, 2),
              ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group, act_type=act_type)])
        self.lower_res1_branch = ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group,
                                               act_type=act_type)
        # pooling 16
        self.lower_res2_main = nn.Sequential(
            *[nn.MaxPool2d(2, 2),
              ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group, act_type=act_type)])
        self.lower_res2_branch = ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group,
                                               act_type=act_type)
        # pooling 8
        self.lower_res3_main = nn.Sequential(
            *[nn.MaxPool2d(2, 2),
              ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group, act_type=act_type)])
        self.lower_res3_branch = ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group,
                                               act_type=act_type)
        # pooling 4
        self.lower_res4_main = nn.Sequential(
            *[nn.MaxPool2d(2, 2),
              ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group, act_type=act_type)])
        # middle
        self.middle = ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group,
                                    act_type=act_type)
        # Upper 8
        self.upper_res4_main = nn.Sequential(
            *[ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group, act_type=act_type),
              nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)])
        # Upper 16
        self.upper_res3_main = nn.Sequential(
            *[ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group, act_type=act_type),
              nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)])
        # Upper 32
        self.upper_res2_main = nn.Sequential(
            *[ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group, act_type=act_type),
              nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)])
        # Upper 64
        self.upper_res1_main = nn.Sequential(
            *[ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group, act_type=act_type),
              nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)])
        # end_layer
        self.output = nn.Sequential(
            ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group, act_type=act_type),
            *optUnit(norm_type=norm_type, act_type=act_type, out_ch=channels))
        # )

    def forward(self, x):
        branch0 = self.lower_res0_branch(x)
        body = self.lower_res1_main(x)
        branch1 = self.lower_res1_branch(body)
        body = self.lower_res2_main(body)
        branch2 = self.lower_res2_branch(body)
        body = self.lower_res3_main(body)
        branch3 = self.lower_res3_branch(body)
        body = self.lower_res4_main(body)
        body = self.middle(body)
        body = self.upper_res4_main(body)
        body = body + branch3
        body = self.upper_res3_main(body)
        body = body + branch2
        body = self.upper_res2_main(body)
        body = body + branch1
        body = self.upper_res1_main(body)
        body = body + branch0
        body = self.output(body)
        return body


class HourglassBlock1(nn.Module):
    """
        Channels and Resolutions Consistent
    """

    def __init__(self, channels, norm_type='BN', act_type='prelu', num_group=None):
        super(HourglassBlock1, self).__init__()
        # Lower 64
        self.lower_res0_branch = ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group,
                                               act_type=act_type)
        # pooling1 32
        self.lower_res1_main = nn.Sequential(
            *[nn.MaxPool2d(2, 2),
              ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group, act_type=act_type)])
        self.lower_res1_branch = ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group,
                                               act_type=act_type)
        # pooling 16
        self.lower_res2_main = nn.Sequential(
            *[nn.MaxPool2d(2, 2),
              ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group, act_type=act_type)])
        self.lower_res2_branch = ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group,
                                               act_type=act_type)
        # middle
        self.middle = ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group,
                                    act_type=act_type)
        # Upper 32
        self.upper_res2_main = nn.Sequential(
            *[ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group, act_type=act_type),
              nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)])
        # Upper 64
        self.upper_res1_main = nn.Sequential(
            *[ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group, act_type=act_type),
              nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)])
        # end_layer
        self.output = nn.Sequential(
            ResidualBlock(channels, channels, 1, 3, norm_type=norm_type, num_group=num_group, act_type=act_type),
            *optUnit(norm_type=norm_type, act_type=act_type, out_ch=channels))

    def forward(self, x):
        branch0 = self.lower_res0_branch(x)
        body = self.lower_res1_main(x)
        branch1 = self.lower_res1_branch(body)
        body = self.lower_res2_main(body)
        branch2 = self.lower_res2_branch(body)
        body = self.middle(body)
        body = body + branch2
        body = self.upper_res2_main(body)
        body = body + branch1
        body = self.upper_res1_main(body)
        body = body + branch0
        body = self.output(body)
        return body


class FeatureMapFusion(nn.Module):

    def __init__(self, channles, down_resolution=None, norm_type='BN', act_type='prelu', num_group=None):
        super().__init__()
        self.down_resolution = down_resolution
        self.neck = nn.Sequential(*(
                optUnit(norm_type=norm_type, act_type=act_type, out_ch=channles + 13) + optUnit(
            opt_type='conv', in_ch=channles + 13, out_ch=channles, ker_size=1, stride=1)))
        self.hourglass = HourglassBlock1(channles, norm_type=norm_type, act_type=act_type, num_group=num_group)
        self.foot = nn.Sequential(*optUnit(norm_type=norm_type, out_ch=channles * 2, num_group=num_group))

    def forward(self, data, pred_heatmap):
        pred_heatmap = F.interpolate(pred_heatmap, scale_factor=1 / self.down_resolution, mode='bilinear',
                                     align_corners=True) if self.down_resolution else pred_heatmap
        body = torch.cat([data, pred_heatmap], 1)
        body = self.neck(body)
        body = self.hourglass(body)
        body = torch.sigmoid(body)
        body = torch.mul(body, data)
        body = torch.cat([body, data], 1)
        body = self.foot(body)
        return body


class MPIBlock(nn.Module):
    """
        Message Passing Layer
        Bi-directional Tree
        B, 16, 64, 64
        有融合且要传递给下一层时需要CBAC - 8 16
    """

    def __init__(self, hourglass_channels, boundaries):
        super(MPIBlock, self).__init__()
        # B, 256, 64, 64 ->B, 13, 64, 64
        self.up = nn.Sequential(
            *optUnit(opt_type='conv', in_ch=hourglass_channels * boundaries, out_ch=16 * boundaries, ker_size=7,
                     stride=1, conv_groups=boundaries))
        # up 1
        self.up1 = nn.Sequential(*optUnit(norm_type='BN', act_type="prelu", out_ch=16))

        # up 2
        self.up1to2 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up2 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))

        # up 3
        self.up2to3 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up3 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))

        # up 4
        self.up1to4 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up3to4 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up4 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))

        # up 5
        self.up2to5 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up4to5 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up5 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))

        # up 6
        self.up1to6 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up5to6 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up6 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))

        # up 7
        self.up1to7 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up6to7 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up7 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))

        # up 8 and 11
        self.up1to8 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up7to8 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up8 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))
        self.up1to11 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up7to11 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up11 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))

        # up 9 and 12
        self.up7to9 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up8to9 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up9 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))
        self.up7to12 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up11to12 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up12 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))

        # up 10 and 13
        self.up7to10 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up9to10 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up10 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))
        self.up7to13 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up12to13 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.up13 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))
        self.up_foot = nn.Sequential(
            *optUnit(opt_type='conv', in_ch=16 * boundaries, out_ch=hourglass_channels, ker_size=1, stride=1))

        # down
        self.down = nn.Sequential(
            *optUnit(opt_type='conv', in_ch=hourglass_channels * boundaries, out_ch=16 * boundaries, ker_size=7,
                     stride=1, conv_groups=boundaries))
        # down 13 and 10
        self.down13 = nn.Sequential(*optUnit(norm_type='BN', act_type="prelu", out_ch=16))
        self.down10 = nn.Sequential(*optUnit(norm_type='BN', act_type="prelu", out_ch=16))

        # down 9 and 12
        self.down10to9 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down9 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))
        self.down13to12 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down12 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))

        # down 8 and 11
        self.down9to8 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down8 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))
        self.down12to11 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down11 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))

        # down 7
        self.down9to7 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down10to7 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down13to7 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down12to7 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down8to7 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down11to7 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down7 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))

        # down 6
        self.down7to6 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))

        self.down6 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))

        # down 5
        self.down6to5 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down5 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))

        # down 4
        self.down5to4 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down4 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))

        # down 3
        self.down4to3 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down3 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))

        # down 2
        self.down3to2 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down5to2 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down2 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))

        # down 1
        self.down2to1 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down4to1 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down6to1 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down7to1 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down8to1 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down11to1 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type='BN', act_type='prelu', in_ch=16, out_ch=8, ker_size=7,
                        stride=1) + optUnit(opt_type='conv', in_ch=8, out_ch=16, ker_size=7, stride=1)))
        self.down1 = nn.Sequential(*optUnit(norm_type='BN', act_type='prelu', out_ch=16))
        self.down_foot = nn.Sequential(
            *optUnit(opt_type='conv', in_ch=16 * boundaries, out_ch=hourglass_channels, ker_size=1, stride=1))


    def forward(self, x):
        # Input(B, 256, 64, 64)
        body = x.repeat((1, 13, 1, 1))
        tree_up = self.up(body)
        # up 1
        up_1 = self.up1(tree_up[:, :16])
        # up 2
        up_1to2 = self.up1to2(up_1)
        up_2 = self.up2(up_1to2 + tree_up[:, 16:16 * 2])
        # up 3
        up_2to3 = self.up2to3(up_2)
        up_3 = self.up3(up_2to3 + tree_up[:, 16 * 2:16 * 3])
        # up 4
        up_1to4 = self.up1to4(up_1)
        up_3to4 = self.up3to4(up_3)
        up_4 = self.up4(up_1to4 + up_3to4 + tree_up[:, 16 * 3:16 * 4])
        # up 5
        up_2to5 = self.up2to5(up_2)
        up_4to5 = self.up4to5(up_4)
        up_5 = self.up5(up_2to5 + up_4to5 + tree_up[:, 16 * 4:16 * 5])
        # up 6
        up_1to6 = self.up1to6(up_1)
        up_5to6 = self.up5to6(up_5)
        up_6 = self.up6(up_1to6 + up_5to6 + tree_up[:, 16 * 5:16 * 6])
        # up 7
        up_1to7 = self.up1to7(up_1)
        up_6to7 = self.up7to8(up_6)
        up_7 = self.up7(up_1to7 + up_6to7 + tree_up[:, 16 * 6:16 * 7])
        # up 8 and 11
        up_1to8 = self.up1to8(up_1)
        up_7to8 = self.up7to8(up_7)
        up_8 = self.up8(up_1to8 + up_7to8 + tree_up[:, 16 * 7:16 * 8])
        up_1to11 = self.up1to11(up_1)
        up_7to11 = self.up7to11(up_7)
        up_11 = self.up11(up_1to11 + up_7to11 + tree_up[:, 16 * 10:16 * 11])
        # up 9 and 12
        up_7to9 = self.up7to9(up_7)
        up_8to9 = self.up8to9(up_8)
        up_9 = self.up9(up_7to9 + up_8to9 + tree_up[:, 16 * 8:16 * 9])
        up_7to12 = self.up7to12(up_7)
        up_11to12 = self.up11to12(up_11)
        up_12 = self.up12(up_7to12 + up_11to12 + tree_up[:, 16 * 11:16 * 12])
        # up 10 and 13
        up_7to10 = self.up7to10(up_7)
        up_9to10 = self.up9to10(up_9)
        up_10 = self.up10(up_7to10 + up_9to10 + tree_up[:, 16 * 9:16 * 10])
        up_7to13 = self.up7to13(up_7)
        up_12to13 = self.up12to13(up_12)
        up_13 = self.up13(up_7to13 + up_12to13 + tree_up[:, 16 * 12:16 * 13])
        up_cat = torch.cat([up_1, up_2, up_3, up_4, up_5, up_6, up_7, up_8, up_9, up_10, up_11, up_12, up_13], 1)
        up_final = self.up_foot(up_cat)

        # down
        tree_down = self.down(body)
        # down 13 and 10
        down_13 = self.up1(tree_down[:, 16 * 12:16 * 13])
        down_10 = self.up1(tree_down[:, 16 * 9:16 * 10])
        # down 9 and 12
        down_10to9 = self.down10to9(down_10)
        down_9 = self.down9(down_10to9 + tree_down[:, 16 * 8:16 * 9])
        down_13to12 = self.down13to12(down_13)
        down_12 = self.down12(down_13to12 + tree_down[:, 16 * 11:16 * 12])
        # down 8 and 11
        down_9to8 = self.down9to8(down_9)
        down_8 = self.down9(down_9to8 + tree_down[:, 16 * 7:16 * 8])
        down_12to11 = self.down12to11(down_12)
        down_11 = self.down12(down_12to11 + tree_down[:, 16 * 10:16 * 11])
        # down 7
        down_9to7 = self.down9to7(down_9)
        down_10to7 = self.down10to7(down_10)
        down_13to7 = self.down13to7(down_13)
        down_12to7 = self.down12to7(down_12)
        down_8to7 = self.down8to7(down_8)
        down_11to7 = self.down11to7(down_11)
        down_7 = self.up4(
            down_9to7 + down_10to7 + down_13to7 + down_12to7 + down_8to7 + down_11to7 + tree_up[:, 16 * 6:16 * 7])
        # down 6
        down_7to6 = self.down7to6(down_7)
        down_6 = self.down6(down_7to6 + tree_up[:, 16 * 5:16 * 6])
        # down 5
        down_6to5 = self.down6to5(down_6)
        down_5 = self.down5(down_6to5 + tree_up[:, 16 * 4:16 * 5])
        # down 4
        down_5to4 = self.down5to4(down_5)
        down_4 = self.down4(down_5to4 + tree_up[:, 16 * 3:16 * 4])
        # down 3
        down_4to3 = self.down4to3(down_4)
        down_3 = self.down3(down_4to3 + tree_up[:, 16 * 2:16 * 3])
        # down 2
        down_3to2 = self.down3to2(down_3)
        down_5to2 = self.down5to2(down_5)
        down_2 = self.down2(down_3to2 + down_5to2 + tree_up[:, 16 * 1:16 * 2])
        # down 1
        down_2to1 = self.down2to1(down_2)
        down_4to1 = self.down4to1(down_4)
        down_6to1 = self.down6to1(down_6)
        down_7to1 = self.down7to1(down_7)
        down_8to1 = self.down8to1(down_8)
        down_11to1 = self.down11to1(down_11)
        down_1 = self.down1(down_2to1 + down_4to1 + down_6to1 + down_7to1 + down_8to1 + down_11to1 + tree_up[:, :16])
        down_cat = torch.cat(
            [down_1, down_2, down_3, down_4, down_5, down_6, down_7, down_8, down_9, down_10, down_11, down_12, down_13], 1)
        down_final = self.down_foot(down_cat)
        output = up_final + down_final
        return output
