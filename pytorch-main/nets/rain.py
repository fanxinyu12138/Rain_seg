import torch
from torch import nn, Tensor
from torch.nn import functional as F


from model.PSA import PolarizedSelfAttention
from model.DMFM import MFM

from backbone.mobilenetv3 import mobilenet_v3_large
from backbone.edgevit import edgevit_xs

from model.DeformConv2d import DeformConv2d
from model.RAM import ChannelAtt

from model.AMACA import AMACA


class MEbackbone(nn.Module):
    def __init__(self):
        super(MEbackbone, self).__init__()
        #.....mobilenet_v3_large
        #.....edgevit_xs
        #.....DeformConv2d
        #.....ChannelAtt

    def forward(self, x):
        #.....

        return


class Rain(nn.Module):
    def __init__(self, num_classes=2, backbone="mobilenet"):
        super(Rain, self).__init__()
        if backbone == "mobilenet":
            self.backbone = MEbackbone()
            in_channels = 256
            mid_features = 40
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet.'.format(backbone))

        self.aspp = AMACA(dim_in=in_channels, dim_out=256, rate=1)

        self.att2 = PolarizedSelfAttention(mid_features)
        self.shortcut_conv1 = nn.Sequential(
            nn.Conv2d(mid_features, 80, 1),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True)
        )
        self.shortcut_conv1_1 = nn.Sequential(
            nn.Conv2d(80, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.att3 = PolarizedSelfAttention(low_level_channels)
        self.shortcut_conv2 = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(40 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )

        self.cat_conv1 = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.bga = MFM(num_classes)

        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)

        low_level_features, mid_features, x = self.backbone(x)

        x = self.aspp(x)

        mid_features = self.att2(mid_features)
        mid_features = self.shortcut_conv1(mid_features)
        mid_features = self.shortcut_conv1_1(mid_features)

        low_level_features = self.att3(low_level_features)
        low_level_features = self.shortcut_conv2(low_level_features)

        x1 = F.interpolate(x, size=(mid_features.size(2), mid_features.size(3)),
                             mode='bilinear', align_corners=True)
        x2 = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)),
                             mode='bilinear', align_corners=True)

        x_d = torch.cat((low_level_features, x2), dim=1)
        x_s = torch.cat((mid_features, x1), dim=1)

        x = self.bga(x_d, x_s)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        return x


rain = Rain()
print(rain)
