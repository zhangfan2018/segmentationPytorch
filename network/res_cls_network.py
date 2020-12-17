
"""Implementation of the ResNet in torch.
Author: zhangfan
Email: zf2016@mail.ustc.edu.cn
data: 2020/12/15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.blocks.residual_blocks import ResConvBatchNormBlock


class ResNet(nn.Module):
    """UNet network"""
    def __init__(self, net_args={"num_class": 1,
                                 "nb_filter": [8, 16, 32, 64, 128]}):
        super().__init__()

        # ResNet parameter.
        num_class = net_args["num_class"]
        nb_filter = net_args["nb_filter"]

        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.conv0_0 = ResConvBatchNormBlock(1, nb_filter[0], nb_filter[0])
        self.conv1_0 = ResConvBatchNormBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = ResConvBatchNormBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = ResConvBatchNormBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = ResConvBatchNormBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv_cls = nn.Sequential(
            ResConvBatchNormBlock(nb_filter[4], nb_filter[4], nb_filter[4]),
            nn.AdaptiveMaxPool3d((1, 1, 1))
        )
        self.fc_cls = torch.nn.Linear(in_features=nb_filter[4], out_features=num_class)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        features_cls = self.conv_cls(self.pool(x4_0))
        features_cls = features_cls.view(features_cls.size()[0], -1)
        out_cls = F.sigmoid(self.fc_cls(features_cls))

        return out_cls


if __name__ == '__main__':
    data = torch.randn([1, 1, 64, 64, 64]).float().cuda()

    model = ResNet().cuda()

    with torch.no_grad():
        out_cls = model(data)

    print(out_cls.shape)