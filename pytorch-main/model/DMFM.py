import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, relu=True):
        super(ConvBnReLU, self).__init__()
        self.add_module(
            'Conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
        )
        self.add_module('BN', nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.999))
        if relu:
            self.add_module('ReLU', nn.ReLU())

class DWConvBnReLU(nn.Sequential):
    def __init__(self,in_channels, out_channels, kernel_size1, kernel_size2, stride, padding1, padding2, dilation=1, groups=1, relu=True):
        super(DWConvBnReLU, self).__init__()
        self.add_module(
            'DWConv', nn.Conv2d(in_channels, in_channels, kernel_size1, stride, padding1, dilation, groups, bias=False),
        )
        self.add_module('BN', nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.999)
        )
        self.add_module('Conv',nn.Conv2d(in_channels,out_channels, kernel_size2, stride, padding2, bias=False)
        )
        if relu:
            self.add_module('ReLU', nn.ReLU()
        )

class ImagePool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.conv = ConvBnReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, x):
        x_size = x.shape
        x = self.pool(x)
        x = self.conv(x)
        x = F.interpolate(x, size=x_size[2:], mode='bilinear', align_corners=True)
        return x


class DMAM(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(DMAM, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module('c0', ConvBnReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1))   # channel->out_channels, keep size

        for idx, rate in enumerate(rates):
            self.stages.add_module('c{}'.format(idx+1), DWConvBnReLU(in_channels, out_channels, kernel_size1=3, kernel_size2=1, stride=1, padding1=rate, groups=in_channels, padding2=0, dilation=rate))   # channel->out_channels, keep size
        self.stages.add_module('imagepool', ImagePool(in_channels, out_channels))    # channel->out_channels, keep size
        self.stages.add_module('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x):

        x = torch.cat([stage(x) for stage in self.stages.children()], dim=1)

        return x


class MFM(nn.Module):
    def __init__(self, num_classes=2):
        super(MFM, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(
                304, 304, kernel_size=3, stride=1,
                padding=1, groups=304, bias=False),
            nn.BatchNorm2d(304),
            nn.Conv2d(
                304, 304, kernel_size=1, stride=1,
                padding=0, bias=False),
        )

        self.left2 = nn.Sequential(
            # nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )

        self.right = nn.Sequential(
            nn.Conv2d(
                304, 304, kernel_size=3, stride=1,
                padding=1, groups=304, bias=False),
            nn.BatchNorm2d(304),
            nn.Conv2d(
                304, 304, kernel_size=1, stride=1,
                padding=0, bias=False),
        )

        self.conv = nn.Sequential(
                    ConvBnReLU(608, 256, kernel_size=1, stride=1, padding=1),  # 图片尺寸不变
                    ConvBnReLU(256, 256, kernel_size=1, stride=1, padding=1),
                    nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]  # 144、144

        left1 = self.left(x_d)  # 304*144*144
        left2 = self.left(x_d)  # 304*144*144
        left2 = self.left2(left2)  # 304*36*36(池化：下采样两倍，这样下面的相乘不匹配，在这里进行4倍下采样

        right1 = self.right(x_s)  # 304*36*36
        right2 = self.right(x_s)  # 304*36*36
        right1 = F.interpolate(right1, size=dsize, mode='bilinear', align_corners=True)  # 304*144*144

        left = left1 * torch.sigmoid(right1)  # 304*144*144
        right = left2 * torch.sigmoid(right2)

        right = F.interpolate(right, size=dsize, mode='bilinear', align_corners=True)

        out = self.conv(torch.cat((right, left), dim=1))

        return out
