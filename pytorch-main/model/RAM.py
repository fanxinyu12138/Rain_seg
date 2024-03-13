import torch
import torch.nn as nn

from SoftPool import soft_pool2d, SoftPool2d


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelAtt(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=2, pool_types=['avg', 'max','soft']):
        super(ChannelAtt, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU()
            #nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
        self.incr = nn.Linear(gate_channels // reduction_ratio, gate_channels)

    def forward(self, x):
        channel_att_sum = None
        # avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # avgpoolmlp = self.mlp(avg_pool)
        # maxpoolmlp=self.mlp(max_pool)
        # pooladd = avgpoolmlp+maxpoolmlp

        self.pool = SoftPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        soft_pool = self.mlp(self.pool(x))

        # soft_pool = self.mlp(x)
        # weightPool = soft_pool * pooladd
        weightPool = soft_pool
        # channel_att_sum = self.mlp(weightPool)
        channel_att_sum = self.incr(weightPool)
        Att = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return Att
