"""Implementation of combination module part of UNet.
"""

import torch
import torch.nn as nn

from network.blocks.residual_blocks import ResFourLayerConvBlock


class CombinationModule(nn.Module):
    """Combination of low-level and high-level features"""
    def __init__(self, low_channels, high_channels):
        super(CombinationModule, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.combinator = ResFourLayerConvBlock(low_channels+high_channels, low_channels, low_channels)

    def forward(self, x_low, x_high):
        x = self.combinator(torch.cat([x_low, self.up(x_high)], 1))

        return x
