"""Multi Residual Block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class conv3d_bn(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=0, activation=True):
        super(conv3d_bn, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv3d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=False),
            nn.InstanceNorm3d(out_channel)
        )
        self.activation = activation

    def forward(self, input):
        out = self.conv_bn(input)
        if self.activation:
            out = F.leaky_relu(out)
        return out


class trans_conv3d_bn(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=2, padding=0):
        super(trans_conv3d_bn, self).__init__()
        self.trans_conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False),
            nn.InstanceNorm3d(out_channel)
        )

    def forward(self, input):
        return self.trans_conv(input)


class MultiResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, alpha=1):
        super(MultiResBlock, self).__init__()
        W = alpha * out_channel
        self.shortcut = conv3d_bn(in_channel=in_channel,
                                  out_channel=int(W * 1),
                                  kernel_size=1,
                                  activation=False
                                  )
        self.conv3 = conv3d_bn(in_channel=in_channel,
                               out_channel=int(W * 1),
                               kernel_size=3,
                               padding=1,
                               activation=True
                              )
        self.conv5 = conv3d_bn(in_channel=int(W * 1),
                               out_channel=int(W * 1),
                               kernel_size=3,
                               padding=1,
                               activation=True
                              )

        self.conv7 = conv3d_bn(in_channel=int(W * 1),
                               out_channel=int(W * 1),
                               kernel_size=3,
                               padding=1,
                               activation=True
                              )
        self.conv_compress = conv3d_bn(in_channel=int(W * 3),
                               out_channel=int(W * 1),
                               kernel_size=1,
                               activation=False
                              )
        self.gn1 = nn.InstanceNorm3d(int(W))
        self.relu = nn.LeakyReLU()

        self.gn2 = nn.InstanceNorm3d(int(W))

    def forward(self, input):
        shortcut = self.shortcut(input)
        out_3 = self.conv3(input)
        out_5 = self.conv5(out_3)
        out_7 = self.conv7(out_5)

        out = torch.cat([out_3, out_5, out_7], 1)
        out = self.conv_compress(out)
        out = self.gn1(out)
        out += shortcut
        out = self.gn2(self.relu(out))
        return out


class ResBlock(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.shortcut = conv3d_bn(in_channel=in_channel,
                                  out_channel=out_channel,
                                  kernel_size=1,
                                  activation=False
                                  )
        self.conv_1 = conv3d_bn(in_channel=in_channel,
                               out_channel=out_channel,
                               kernel_size=3,
                               padding=1,
                               activation=True
                              )
        self.gn = nn.InstanceNorm3d(out_channel)
        self.relu = nn.LeakyReLU()

    def forward(self, input):
        shortcut = self.shortcut(input)
        out = self.conv_1(input)
        out += shortcut
        out = self.gn(self.relu(out))
        return out


class ResPath(nn.Module):
    def __init__(self, in_channel, out_channel, length):
        super(ResPath, self).__init__()
        self.res_block_1 = ResBlock(in_channel, out_channel)
        self.res_blocks = self._make_layer(ResBlock(out_channel, out_channel), length)

    def _make_layer(self, block, length):
        layers = []
        for i in range(length - 1):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, input):
        out = self.res_block_1(input)
        out = self.res_blocks(out)
        return out


class MutiResUnet3D(nn.Module):
    def __init__(self, input_channels=1, first_channel=8, num_classes=1):
        super(MutiResUnet3D, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=1)
        self.up = torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.mresblock1 = MultiResBlock(input_channels, first_channel)
        self.res1 = ResPath(first_channel, first_channel, 3)

        self.mresblock2 = MultiResBlock(first_channel, first_channel*2)
        self.res2 = ResPath(first_channel*2, first_channel*2, 2)

        self.mreblock3 = MultiResBlock(first_channel*2, first_channel*4)
        self.res3 = ResPath(first_channel*4, first_channel*4, 1)

        self.mreblock4 = MultiResBlock(first_channel*4, first_channel*8)

        self.up1 = MultiResBlock(first_channel*12, first_channel*4)
        self.up2 = MultiResBlock(first_channel*6, first_channel*2)
        self.up3 = MultiResBlock(first_channel*3, first_channel)

        self.out = nn.Conv3d(first_channel, num_classes, kernel_size=1)

    def forward(self, input):
        x_0 = self.mresblock1(input)
        pool_0 = self.pool(x_0)
        x_0 = self.res1(x_0)

        x_1 = self.mresblock2(pool_0)
        pool_1 = self.pool(x_1)
        x_1 = self.res2(x_1)

        x_2 = self.mreblock3(pool_1)
        pool_2 = self.pool(x_2)
        x_2 = self.res3(x_2)

        x_3 = self.mreblock4(pool_2)

        up_1 = torch.cat([x_2, F.interpolate(x_3, size=(x_2.size(2),
                          x_2.size(3), x_2.size(4)), mode='trilinear', align_corners=True)], 1)
        y_1 = self.up1(up_1)

        up_2 = torch.cat([x_1, F.interpolate(y_1, size=(x_1.size(2),
                          x_1.size(3), x_1.size(4)), mode='trilinear', align_corners=True)], 1)
        y_2 = self.up2(up_2)

        up_3 = torch.cat([x_0, F.interpolate(y_2, size=(x_0.size(2),
                          x_0.size(3), x_0.size(4)), mode='trilinear', align_corners=True)], 1)
        y_3 = self.up3(up_3)

        out = self.out(y_3)

        return [out]


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = MutiResUnet3D().to(device)
    x = torch.ones([1, 1, 64, 64, 64]).to(device)
    y = net(x)
    print(y)