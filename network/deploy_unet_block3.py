
"""Deployment of end-to-end UNet in torchscript,
which is compose of preprocess, segmentation procedure and postprocess.
Author: zhangfan
Email: zf2016@mail.ustc.edu.cn
data: 2020/10/14
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.blocks.residual_blocks import ResFourLayerConvBlock


class Input(nn.Module):
    """Input layer, including re-sample, clip and normalization image."""
    def __init__(self, input_size=(224, 160, 224), clip_window=(-1200, 1200)):
        super(Input, self).__init__()
        self.input_size = input_size
        self.clip_window = clip_window

    def forward(self, x):
        x = F.interpolate(x, size=self.input_size, mode='trilinear', align_corners=True)
        x = torch.clamp(x, min=self.clip_window[0], max=self.clip_window[1])
        mean = torch.mean(x)
        std  = torch.std(x)
        x = (x - mean) / (1e-5 + std)

        return x


class Output(nn.Module):
    """Output layer, re-sample image to original size."""
    def __init__(self):
        super(Output, self).__init__()

    def forward(self, x, x_input):
        x = F.interpolate(x, size=(x_input.size(2), x_input.size(3), x_input.size(4)), mode='trilinear', align_corners=True)
        # x = F.sigmoid(x)

        return x


class UNet(nn.Module):
    """UNet network"""
    def __init__(self, net_args={"num_class": 1,
                                 "nb_filter": [12, 24, 48, 96],
                                 "input_size": [224, 160, 224],
                                 "clip_window": [-1200, 1200]}):
        super().__init__()

        # UNet parameter.
        num_class = net_args["num_class"]
        nb_filter = net_args["nb_filter"]

        self.input = Input(input_size=net_args["input_size"], clip_window=net_args["clip_window"])
        self.output = Output()

        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv0_0 = ResFourLayerConvBlock(1, nb_filter[0], nb_filter[0])
        self.conv1_0 = ResFourLayerConvBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = ResFourLayerConvBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = ResFourLayerConvBlock(nb_filter[2], nb_filter[3], nb_filter[3])

        self.conv2_2 = ResFourLayerConvBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = ResFourLayerConvBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = ResFourLayerConvBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv3d(nb_filter[0], num_class, kernel_size=1, bias=False)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_input):
        x = self.input(x_input)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        output = self.output(output, x_input)

        return output


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    data = torch.randn([1, 1, 512, 512, 512]).float().cuda()

    model = UNet().cuda()

    with torch.no_grad():
        output = model(data)

    print(output.shape)