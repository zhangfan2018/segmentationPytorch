"""Implementation of UNet in separate modules.
"""

import torch
import torch.nn as nn

from network.unet_framework.encoder import EncoderModule
from network.unet_framework.decoder import DecoderModule


class UNet(nn.Module):
    """UNet is compose of encoder and decoder module."""
    def __init__(self, in_channel=1, num_filters=[8, 16, 32, 64, 128], out_channel=1):
        super(UNet, self).__init__()
        self.encoder_module = EncoderModule(in_channel=in_channel, num_filters=num_filters)
        self.decoder_module = DecoderModule(num_filters=num_filters, out_channel=out_channel)

    def forward(self, x):
        features = self.encoder_module(x)
        out = self.decoder_module(features)

        return out


if __name__ == '__main__':
    data = torch.randn([1, 1, 32, 32, 32]).float().cuda()
    model = UNet().cuda()

    with torch.no_grad():
        predict = model(data)
    print(predict.shape)
