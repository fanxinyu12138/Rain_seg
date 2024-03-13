import torch
import torch.nn as nn
import torch.nn.functional as F

from model.PSA import PolarizedSelfAttention

class ConvBnReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, relu=True):
        super(ConvBnReLU, self).__init__()
        self.add_module(
            'Conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
        )
        self.add_module('BN', nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.999))
        if relu:
            self.add_module('ReLU', nn.ReLU())

class ImagePool_1(nn.Module):
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

# DCN
class AMACA(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(AMACA, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 1, 1, padding=0, dilation=rate, bias=True, groups=dim_in),
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=1, bias=True),

            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, 1, padding=4 * rate, dilation=4 * rate, bias=True, groups=dim_in),
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=1, bias=True),


            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, 1, padding=8 * rate, dilation=8 * rate, bias=True, groups=dim_in),
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=1, bias=True),


            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True, groups=dim_in),
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=1, bias=True),

            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch6 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, 1, padding=16 * rate, dilation=16 * rate, bias=True, groups=dim_in),
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=1, bias=True),

            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.maxpo = ImagePool_1(dim_in, dim_out)

        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.att1 = PolarizedSelfAttention(dim_out*7)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 7, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()

        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        conv3x3_4 = self.branch6(x)

        global_feature = self.maxpo(x)

        feature_cat = torch.cat([x, conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, conv3x3_4, global_feature], dim=1)

        feature_cat = self.att1(feature_cat)

        result = self.conv_cat(feature_cat)

        return result
