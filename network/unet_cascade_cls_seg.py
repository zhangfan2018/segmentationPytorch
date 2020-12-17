
"""Implementation of the Unet in torch.
Author: zhangfan
Email: zf2016@mail.ustc.edu.cn
data: 2020/09/09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.blocks.residual_blocks import ResConvBatchNormBlock


class UNetMultiTask(nn.Module):
    """UNet network"""
    def __init__(self, net_args={"num_class": 1,
                                 "nb_filter": [8, 16, 32, 64, 128]}):
        super().__init__()

        # UNet parameter.
        num_class = net_args["num_class"]
        nb_filter = net_args["nb_filter"]

        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv0_0 = ResConvBatchNormBlock(1, nb_filter[0], nb_filter[0])
        self.conv1_0 = ResConvBatchNormBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = ResConvBatchNormBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = ResConvBatchNormBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = ResConvBatchNormBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = ResConvBatchNormBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = ResConvBatchNormBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = ResConvBatchNormBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = ResConvBatchNormBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.conv_cls = nn.Sequential(
            ResConvBatchNormBlock(1, nb_filter[0], nb_filter[0]),
            nn.AdaptiveMaxPool3d((1, 1, 1))
        )
        self.fc_cls = torch.nn.Linear(in_features=nb_filter[0], out_features=num_class)

        self.conv_seg = nn.Conv3d(nb_filter[0], num_class, kernel_size=1, bias=False)

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

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        out_seg = self.conv_seg(x0_4)
        features_cls = self.conv_cls(self.pool(out_seg))
        features_cls = features_cls.view(features_cls.size()[0], -1)
        out_cls = F.sigmoid(self.fc_cls(features_cls))

        return [out_cls], [out_seg]


if __name__ == '__main__':
    data = torch.randn([1, 1, 64, 64, 64]).float().cuda()

    model = UNetMultiTask().cuda()

    with torch.no_grad():
        out_cls, out_seg = model(data)

    print(out_cls[0].shape)
    print(out_seg[0].shape)