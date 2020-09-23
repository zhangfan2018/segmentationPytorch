"""Split-Attention"""

import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['SplAtConv3d']

class SplAtConv3d(nn.Module):
    """Split-Attention Conv3d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True,
                 radix=2, norm_layer="GN",
                 dropblock_prob=0.0):
        super(SplAtConv3d, self).__init__()
        inter_channels = max(in_channels*radix//2, 8)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.conv = nn.Conv3d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias)
        self.bn0 = nn.BatchNorm3d(num_features=channels*radix) if norm_layer=="BN" else \
                   nn.GroupNorm(num_groups=channels // 2, num_channels=channels*radix)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv3d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = nn.BatchNorm3d(num_features=inter_channels) if norm_layer=="BN" else \
                   nn.GroupNorm(num_groups=inter_channels // 4, num_channels=inter_channels)
        self.fc2 = nn.Conv3d(inter_channels, channels*radix, 1, groups=self.cardinality)
        self.dropblock = nn.Dropout(p=dropblock_prob) if dropblock_prob > 0.0 else nn.Sequential()
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool3d(gap, 1)
        gap = self.fc1(gap)

        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x