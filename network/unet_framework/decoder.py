"""Implementation of decoder module part of UNet.
"""

import torch.nn as nn

from network.unet_framework.combinator import CombinationModule


class DecoderModule(nn.Module):
    """Extraction of low-level positional information and high-level semantic information"""
    def __init__(self, num_filters=[8, 16, 32, 64, 128], out_channel=1):
        super(DecoderModule, self).__init__()
        self.num_blocks = len(num_filters) - 1
        for i in range(self.num_blocks):
            decoder_block = CombinationModule(num_filters[self.num_blocks-i-1], num_filters[self.num_blocks-i])
            self.__setattr__("dec_block_"+str(i+1), decoder_block)
        self.final = nn.Conv3d(num_filters[0], out_channel, kernel_size=1, bias=False)

    def forward(self, features):
        x = features[-1]
        for i in range(self.num_blocks):
            x = self.__getattr__("dec_block_"+str(i+1))(features[self.num_blocks-i-1], x)
        x = self.final(x)

        return x
