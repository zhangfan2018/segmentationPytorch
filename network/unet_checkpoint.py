
"""Implementation of the high-efficient GPU memory use of UNet in torch.
Author: zhangfan
Email: zf2016@mail.ustc.edu.cn
data: 2020/09/04
"""

import os
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


class conv_3_check(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_checkpoint=False):
        """convolution of 3*3*3 in checkpoint mode."""
        super(conv_3_check, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        if self.use_checkpoint:
            out = cp.checkpoint(self.conv3, x)
        else:
            out = self.conv3(x)
        return out


class conv_1_check(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_checkpoint=False):
        """convolution of 1*1*1 in checkpoint mode."""
        super(conv_1_check, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        if self.use_checkpoint:
            out = cp.checkpoint(self.conv, x)
        else:
            out = self.conv(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, stride=1, use_checkpoint=False):
        """residual block, including two layer convolution,
        instance normalization  and leaky ReLU"""
        super(ResBlock, self).__init__()
        self.residual_unit = nn.Sequential(
            conv_3_check(in_channel, mid_channel, stride=1, use_checkpoint=use_checkpoint),
            nn.InstanceNorm3d(mid_channel),
            nn.LeakyReLU(inplace=True),
            conv_3_check(mid_channel, out_channel, stride=stride, use_checkpoint=use_checkpoint),
            nn.InstanceNorm3d(out_channel),
        )
        self.shortcut_unit = nn.Sequential(
            conv_1_check(in_channel, out_channel, stride=stride, use_checkpoint=False),
            nn.InstanceNorm3d(out_channel),
        )
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        output = self.residual_unit(x)
        output += self.shortcut_unit(x)
        output = self.leaky_relu(output)

        return output


class UNet(nn.Module):
    """UNet network"""
    def __init__(self, net_args={"num_class": 1,
                                 "nb_filter": [8, 16, 32, 64, 128],
                                 "use_checkpoint": False}):
        super().__init__()

        # UNet parameter.
        num_class = net_args["num_class"] if "num_class" in net_args else 1
        nb_filter = net_args["nb_filter"] if "nb_filter" in net_args else [8, 16, 32, 64, 128]
        use_checkpoint = net_args["use_checkpoint"] if "use_checkpoint" in net_args else False

        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv0_0 = ResBlock(1, nb_filter[0], nb_filter[0], stride=1, use_checkpoint=use_checkpoint)
        self.conv1_0 = ResBlock(nb_filter[0], nb_filter[1], nb_filter[1], stride=2, use_checkpoint=use_checkpoint)
        self.conv2_0 = ResBlock(nb_filter[1], nb_filter[2], nb_filter[2], stride=2, use_checkpoint=use_checkpoint)
        self.conv3_0 = ResBlock(nb_filter[2], nb_filter[3], nb_filter[3], stride=2, use_checkpoint=use_checkpoint)
        self.conv4_0 = ResBlock(nb_filter[3], nb_filter[4], nb_filter[4], stride=2, use_checkpoint=use_checkpoint)

        self.conv3_1 = ResBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = ResBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = ResBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = ResBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv3d(nb_filter[0], num_class, kernel_size=1, bias=False)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)

        return [output]


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    data = torch.randn([1, 1, 128, 128, 128]).float().cuda()

    model = UNet().cuda()

    with torch.no_grad():
        output = model(data)

    print(output.shape)