from torch import nn
import torch
import torch.nn.functional as F
from basic_block import ResidualBlock, HourglassBlock, MPIBlock, FeatureMapFusion
from basic_unit import optUnit

norm_layer = (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)
opt_layer = (nn.Conv2d, nn.Linear)


class PreRes(nn.Module):
    def __init__(self, input_channel, output_channel, norm_type='BN', act_type='prelu', num_group=None):
        super().__init__()
        # 1, 256
        layers = optUnit(opt_type='conv', norm_type=norm_type, act_type=act_type, in_ch=input_channel,
                         out_ch=output_channel // 4, ker_size=7, stride=2, num_group=num_group)
        # 升维 64-128
        layers.append(ResidualBlock(output_channel // 4, output_channel // 2, stride=1, res_type=2, norm_type=norm_type,
                                    act_type=act_type, num_group=num_group))
        # 降分辨率
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(ResidualBlock(output_channel // 2, output_channel // 2, stride=1, res_type=3, norm_type=norm_type,
                                    act_type=act_type, num_group=num_group))
        layers += optUnit(norm_type=norm_type, act_type=act_type, out_ch=output_channel // 2)

        # 升维 128-256
        layers.append(ResidualBlock(output_channel // 2, output_channel, stride=1, res_type=2, norm_type=norm_type,
                                    act_type=act_type, num_group=num_group))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        body = self.body(x)
        return body


class BoundaryHeatmapEstimator(nn.Module):
    def __init__(self, input_channel, hourglass_channels, boundaries, norm_type='BN', act_type='prelu', num_group=None):
        super().__init__()
        # (in, out) = (1, 256)
        self.pre = PreRes(input_channel, hourglass_channels, norm_type=norm_type, act_type=act_type,
                          num_group=num_group)
        self.hourglass1 = HourglassBlock(hourglass_channels, norm_type=norm_type, act_type=act_type,
                                         num_group=num_group)
        self.hourglass2 = HourglassBlock(hourglass_channels, norm_type=norm_type, act_type=act_type,
                                         num_group=num_group)
        self.hourglass3 = HourglassBlock(hourglass_channels, norm_type=norm_type, act_type=act_type,
                                         num_group=num_group)
        self.hourglass4 = HourglassBlock(hourglass_channels, norm_type=norm_type, act_type=act_type,
                                         num_group=num_group)
        self.heatmap1 = nn.Sequential(
            *optUnit(opt_type='conv', in_ch=hourglass_channels, out_ch=boundaries, ker_size=7, stride=1))
        self.heatmap2 = nn.Sequential(
            *optUnit(opt_type='conv', in_ch=hourglass_channels, out_ch=boundaries, ker_size=7, stride=1))
        self.heatmap3 = nn.Sequential(
            *optUnit(opt_type='conv', in_ch=hourglass_channels, out_ch=boundaries, ker_size=7, stride=1))
        self.heatmap4 = nn.Sequential(
            *optUnit(opt_type='conv', in_ch=hourglass_channels, out_ch=boundaries, ker_size=7, stride=1))

        for _layer in self.modules():
            if isinstance(_layer, opt_layer):
                nn.init.kaiming_normal_(_layer.weight, 2 ** 0.5)
            if isinstance(_layer, norm_layer):
                nn.init.constant_(_layer.weight, 1.0)
                nn.init.constant_(_layer.bias, 0.0)

    def forward(self, data, useCompress=False):
        # no message passing version
        pre = self.pre(data / 256)
        hourglass_output1 = self.hourglass1(pre)
        hourglass_output2 = self.hourglass2(hourglass_output1)
        hourglass_output3 = self.hourglass3(hourglass_output2)
        hourglass_output4 = self.hourglass4(hourglass_output3)
        compress = lambda x: (x - torch.min(x, dim=1, keepdim=True)[0]) / (
                torch.max(x, dim=1, keepdim=True)[0] - torch.min(x, dim=1, keepdim=True)[0])
        # compress = lambda x: F.sigmoid(x)
        inter1 = self.heatmap1(hourglass_output1)
        inter2 = self.heatmap2(hourglass_output2)
        inter3 = self.heatmap3(hourglass_output3)
        pred_heatmap = self.heatmap4(hourglass_output4)
        if useCompress:
            return compress(inter1), compress(inter2), compress(inter3), compress(pred_heatmap)
        else:
            return inter1, inter2, inter3, pred_heatmap


class BoundaryHeatmapEstimatorwithMPL(nn.Module):
    def __init__(self, input_channel, hourglass_channels, boundaries, norm_type='BN', act_type='prelu', num_group=None):
        super().__init__()
        # (in, out) = (1, 256)
        self.pre = PreRes(input_channel, hourglass_channels, norm_type=norm_type, act_type=act_type,
                          num_group=num_group)
        self.hourglass1 = HourglassBlock(hourglass_channels, norm_type=norm_type, act_type=act_type,
                                         num_group=num_group)
        self.mpl1 = MPIBlock(hourglass_channels, boundaries)
        self.hourglass2 = HourglassBlock(hourglass_channels, norm_type=norm_type, act_type=act_type,
                                         num_group=num_group)
        self.mpl2 = MPIBlock(hourglass_channels, boundaries)

        self.hourglass3 = HourglassBlock(hourglass_channels, norm_type=norm_type, act_type=act_type,
                                         num_group=num_group)
        self.mpl3 = MPIBlock(hourglass_channels, boundaries)

        self.hourglass4 = HourglassBlock(hourglass_channels, norm_type=norm_type, act_type=act_type,
                                         num_group=num_group)
        self.mpl4 = MPIBlock(hourglass_channels, boundaries)

        self.heatmap = nn.Sequential(
            *optUnit(opt_type='conv', in_ch=hourglass_channels, out_ch=boundaries, ker_size=1, stride=1))

        for _layer in self.modules():
            if isinstance(_layer, opt_layer):
                nn.init.kaiming_normal_(_layer.weight, 2 ** 0.5)
            if isinstance(_layer, norm_layer):
                nn.init.constant_(_layer.weight, 1.0)
                nn.init.constant_(_layer.bias, 0.0)

    def forward(self, data):
        pre = self.pre((data - 127.5) / 256)
        hourglass_output1 = self.hourglass1(pre)
        mpl1 = self.mpl1(hourglass_output1)
        hourglass_output2 = self.hourglass2(mpl1)
        mpl2 = self.mpl2(hourglass_output2)
        hourglass_output3 = self.hourglass3(mpl2 + mpl1)
        mpl3 = self.mpl3(hourglass_output3)
        hourglass_output4 = self.hourglass4(mpl3 + mpl2)
        mpl4 = self.mpl4(hourglass_output4)
        compress = lambda x: (x - torch.min(x, dim=1, keepdim=True)[0]) / (
                torch.max(x, dim=1, keepdim=True)[0] - torch.min(x, dim=1, keepdim=True)[0])
        inter1 = self.heatmap(mpl1)
        inter2 = self.heatmap(mpl2)
        inter3 = self.heatmap(mpl3)
        pred_heatmap = self.heatmap(mpl4)
        # return inter1, inter2, inter3, pred_heatmap
        return compress(inter1), compress(inter2), compress(inter3), compress(pred_heatmap)


class LandmarksRegressor(nn.Module):
    def __init__(self, channels, input_channels=14, norm_type='BN', act_type='prelu',
                 num_group=None):  # input: 14, channels: 256
        super().__init__()
        # B, 14, 256, 256 -> B, 32, 64, 64
        self.res0 = nn.Sequential(*(
                optUnit(opt_type='conv', norm_type=norm_type, act_type=act_type, in_ch=input_channels,
                        out_ch=channels // 16, ker_size=7, stride=2, num_group=num_group) + [
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    ResidualBlock(channels // 16, channels // 8, 1, 2, norm_type=norm_type, act_type=act_type,
                                  num_group=num_group),
                    ResidualBlock(channels // 8, channels // 8, 1, 2, norm_type=norm_type, act_type=act_type,
                                  num_group=num_group)]))
        # B, 32, 64, 64  ->  B, 64, 64, 64  ->  B, 64, 32, 32
        self.fmf1 = FeatureMapFusion(channels // 8, None)
        self.res1 = nn.Sequential(
            ResidualBlock(channels // 4, channels // 4, 2, 2, 4, norm_type=norm_type, act_type=act_type,
                          num_group=num_group),
            ResidualBlock(channels // 4, channels // 4, 1, 3, 4, norm_type=norm_type, act_type=act_type,
                          num_group=num_group))
        # B, 64, 32, 32  ->  B, 128, 32, 32 ->  B, 128, 16, 16
        self.fmf2 = FeatureMapFusion(channels // 4, 2)
        self.res2 = nn.Sequential(
            ResidualBlock(channels // 2, channels // 2, 2, 2, 4, norm_type=norm_type, act_type=act_type,
                          num_group=num_group),
            ResidualBlock(channels // 2, channels // 2, 1, 3, 4, norm_type=norm_type, act_type=act_type,
                          num_group=num_group))
        # B, 128, 16, 16 ->  B, 256, 16, 16 ->  B, 256, 8, 8
        self.fmf3 = FeatureMapFusion(channels // 2, 4)
        self.res3 = nn.Sequential(
            ResidualBlock(channels, channels, 2, 2, 4, norm_type=norm_type, act_type=act_type, num_group=num_group),
            ResidualBlock(channels, channels, 1, 3, 4, norm_type=norm_type, act_type=act_type, num_group=num_group))
        # B, 256, 8, 8 -> B, 256 -> B, 256 -> B, 196
        self.output0 = nn.Conv2d(channels, channels, 8, 1, 0, bias=False)
        self.output1 = nn.Sequential(nn.BatchNorm1d(channels), nn.PReLU(),
                                     nn.Linear(channels, channels, False),
                                     nn.BatchNorm1d(channels),
                                     nn.Dropout2d(p=0.4),
                                     nn.BatchNorm1d(channels),
                                     nn.Linear(channels, 196, False))

        for _layer in self.modules():
            if isinstance(_layer, opt_layer):
                nn.init.xavier_normal_(_layer.weight, 2 ** 0.5)
            if isinstance(_layer, norm_layer):
                nn.init.constant_(_layer.weight, 1.0)
                nn.init.constant_(_layer.bias, 0.0)

    def forward(self, data, pred_heatmap):
        pred_heatmap_256 = F.interpolate(pred_heatmap, scale_factor=4, mode='bilinear', align_corners=True)
        img_fusion = torch.cat([torch.mul(pred_heatmap_256, torch.mean(data, dim=1, keepdim=True)), data], 1)
        body = self.res0(img_fusion)
        body = self.fmf1(body, pred_heatmap)
        body = self.res1(body)
        body = self.fmf2(body, pred_heatmap)
        body = self.res2(body)
        body = self.fmf3(body, pred_heatmap)
        body = self.res3(body)
        body = self.output0(body)
        body = body.view(-1, body.shape[1])
        pred_landmarks = self.output1(body)
        return pred_landmarks


class Discriminator(nn.Module):
    def __init__(self, boundaries, norm_type='IN', act_type='leakyRelu'):
        super().__init__()
        # B, 13, 64, 64 -> B, 14, 64, 64 -> B, 16, 4, 4
        channel = [boundaries + 1, 64, 128, 256, 16]
        layers = optUnit(opt_type='conv', norm_type=norm_type, act_type=act_type, in_ch=channel[0], out_ch=channel[1],
                         ker_size=3, stride=2)  # 32
        layers += optUnit(opt_type='conv', norm_type=norm_type, act_type=act_type, in_ch=channel[1], out_ch=channel[2],
                          ker_size=3, stride=2)  # 16
        layers += optUnit(opt_type='conv', norm_type=norm_type, act_type=act_type, in_ch=channel[2], out_ch=channel[3],
                          ker_size=3, stride=2)  # 8
        layers += optUnit(opt_type='conv', in_ch=channel[3], out_ch=channel[4], ker_size=3, stride=2)  # 16, 4, 4
        self.layers = nn.Sequential(*layers)

        for _layer in self.modules():
            if isinstance(_layer, opt_layer):
                nn.init.kaiming_normal_(_layer.weight, 2 ** 0.5)
            if isinstance(_layer, norm_layer):
                nn.init.constant_(_layer.weight, 1.0)
                nn.init.constant_(_layer.bias, 0.0)

    def forward(self, data, boundary, reality=False):
        data_64 = F.interpolate(data, scale_factor=1 / 4, mode='bilinear', align_corners=True)
        img_fusion = torch.cat([torch.mul(boundary, data_64), data_64], 1)
        output = self.layers(img_fusion)
        return output


class DiscriL2(nn.Module):

    def __init__(self, boundary):
        super().__init__()
        self.conv = nn.Conv2d(boundary, 1, 64, 1, 0, bias=False)

        for _layer in self.modules():
            if isinstance(_layer, nn.Conv2d):
                nn.init.constant_(_layer.weight, 1.0)

    def forward(self, real, pred):
        loss = 0
        for i in range(len(pred)):
            loss += torch.mean(self.conv((real - pred[i]) ** 2))
        # loss += torch.mean(self.conv((real - pred[3]) ** 2))
        return loss / (64 * 64)


if __name__ == "__main__":
    # net = Discriminator(13).cpu()
    # output = net(torch.ones((1, 1, 256, 256)), torch.ones((1, 13, 64, 64)))
    net = DiscriL2(13).cpu()
    output = net(torch.ones((1, 13, 64, 64)))
