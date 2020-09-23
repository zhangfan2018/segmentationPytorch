"""Implementation of the residual block in torch.
Blocks:
ResTwoLayerConvBlock
ResFourLayerConvBlock
ResBottleneckBlock
DepthSeparableConv
"""

import torch
import torch.nn as nn


class ResTwoLayerConvBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, p=0.2, stride=1):
        """residual block, including two layer convolution,
        instance normalization and leaky ReLU"""
        super(ResTwoLayerConvBlock, self).__init__()
        self.residual_unit = nn.Sequential(
            nn.Conv3d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(mid_channel),
            nn.Dropout3d(p=p, inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_channel),
        )
        self.shortcut_unit = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
            nn.InstanceNorm3d(out_channel),
        )
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        output = self.residual_unit(x)
        output += self.shortcut_unit(x)
        output = self.leaky_relu(output)

        return output


class ResFourLayerConvBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, p=0.2):
        """residual block, including four layer convolution,
        instance normalization, drop out and leaky ReLU"""
        super(ResFourLayerConvBlock, self).__init__()
        self.residual_unit_1 = nn.Sequential(
            nn.Conv3d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(mid_channel),
            nn.Dropout3d(p=p, inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(mid_channel),
        )
        self.residual_unit_2 = nn.Sequential(
            nn.Conv3d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(mid_channel),
            nn.Dropout3d(p=p, inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channel),
        )
        self.shortcut_unit_1 = nn.Sequential(
            nn.Conv3d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.InstanceNorm3d(mid_channel),
        )

        self.shortcut_unit_2 = nn.Sequential()
        self.leaky_relu_1 = nn.LeakyReLU(inplace=True)
        self.leaky_relu_2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        output_1 = self.residual_unit_1(x)
        output_1 += self.shortcut_unit_1(x)
        output_1 = self.leaky_relu_1(output_1)
        output_2 = self.residual_unit_2(output_1)
        output_2 += self.shortcut_unit_2(output_1)
        output = self.leaky_relu_2(output_2)

        return output


class ResBottleneckBlock(nn.Module):
    """Residual block in bottleneck structure"""
    def __init__(self, in_channel, mid_channel, out_channel, p=0.2, stride=1, expansion=2):
        super(ResBottleneckBlock, self).__init__()
        self.residual_unit_1 = nn.Sequential(
            nn.Conv3d(in_channel, mid_channel//expansion, kernel_size=1, bias=False),
            nn.InstanceNorm3d(mid_channel//expansion),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channel//expansion, mid_channel//expansion, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(mid_channel//expansion),
            nn.Dropout3d(p=p, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channel//expansion, out_channel, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channel),
        )
        self.shortcut_unit_1 = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
            nn.InstanceNorm3d(out_channel),
        )

        self.relu_1 = nn.ReLU(inplace=True)

    def forward(self, x):
        output_1 = self.residual_unit_1(x)
        output_1 += self.shortcut_unit_1(x)
        output = self.relu_1(output_1)

        return output


class DepthSeparableConv(nn.Module):
    """Depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthSeparableConv, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=stride,
                                   padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


