"""Implementation of encoder module part of UNet.
"""

import torch.nn as nn

from network.blocks.residual_blocks import ResTwoLayerConvBlock, ResFourLayerConvBlock


class EncoderModule(nn.Module):
    """Extraction of hierarchical features by convolution"""
    def __init__(self, in_channel=1, num_filters=[8, 16, 32, 64, 128]):
        super(EncoderModule, self).__init__()
        self.num_blocks = len(num_filters) - 1
        self.block_0 = ResTwoLayerConvBlock(in_channel, num_filters[0], num_filters[0])
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        for i in range(self.num_blocks):
            block = ResFourLayerConvBlock(num_filters[i], num_filters[i+1], num_filters[i+1])
            self.__setattr__("block_"+str(i+1), block)

    def forward(self, x):
        features = []
        x = self.block_0(x)
        features.append(x)

        for i in range(self.num_blocks):
            x = self.__getattr__("block_"+str(i+1))(self.pool(x))
            features.append(x)

        return features
