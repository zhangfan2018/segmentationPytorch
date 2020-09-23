
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.blocks.basic_blocks import conv_kernel_1, adaptive_pool, batch_normal


class Attention_Gate_Block(nn.Module):
    """ Attention Gate Block.

    Reference:
    Ozan Oktay, Jo Schlemper. "Attention U-Net: Learning Where to Look for the Pancreas", MIDL, 2018.

    """
    def __init__(self, F_g, F_l, F_int, model_type="2D"):
        super(Attention_Gate_Block, self).__init__()
        self.W_g = nn.Sequential(conv_kernel_1(F_g, F_int, model_type=model_type, stride=1),
                                 batch_normal(F_int, model_type=model_type))

        self.W_x = nn.Sequential(conv_kernel_1(F_l, F_int, model_type=model_type, stride=1),
                                 batch_normal(F_int, model_type=model_type))

        self.psi = nn.Sequential(conv_kernel_1(F_int, 1, model_type=model_type, stride=1),
                                 batch_normal(1, model_type=model_type),
                                 nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class PSP_Block(nn.Module):
    """Pyramid Scene Parsing Block.

    Reference:
    Hengshuang Zhao, Jianping Shi."Pyramid Scene Parsing Network", CVPR, 2017.

    """
    def __init__(self, features, out_features=256, sizes=(1, 2, 3, 6), model_type="2D"):
        super(PSP_Block, self).__init__()
        self.model_type = model_type
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = conv_kernel_1(features * (len(sizes) + 1), out_features, model_type=model_type)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = adaptive_pool(size=size, pool_type="average", model_type=self.model_type)
        conv = conv_kernel_1(features, features, model_type=self.model_type)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        size = feats.size()[2:]
        priors = [F.interpolate(input=stage(feats), size=size, mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))

        return self.relu(bottle)


class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, normalization="GN"):
        super(_ASPPConv, self).__init__()
        num_groups = out_channels // 4
        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=atrous_rate,
                              dilation=atrous_rate, bias=False)
        # self.normalization = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels) \
        #     if normalization == "GN" else nn.BatchNorm3d(num_features=out_channels)
        self.normalization = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalization(x)
        return self.relu(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, normalization="GN"):
        super(_AsppPooling, self).__init__()
        num_groups = out_channels // 4
        self.gap = nn.Sequential(
                                 nn.AdaptiveAvgPool3d(1),
                                 nn.Conv3d(in_channels, out_channels, 1, bias=False),
                                 # nn.GroupNorm(num_groups=num_groups, num_channels=out_channels) \
                                 # if normalization == "GN" else nn.BatchNorm3d(num_features=out_channels),
                                 nn.InstanceNorm3d(out_channels),
                                 nn.LeakyReLU(inplace=True))

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='trilinear', align_corners=True)
        return out


class ASPP_Block(nn.Module):
    """Atrous Separable Pyramid Parsing Block.

    Reference:
    Liang-Chieh Chen, Yukun Zhu. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation",
    CVPR, 2018.

    """
    def __init__(self, in_channels, out_channels, atrous_rates, normalization="GN"):
        super(ASPP_Block, self).__init__()

        num_groups = out_channels // 4
        self.b0 = nn.Sequential(
                                nn.Conv3d(in_channels, out_channels, 1, bias=False),
                                # nn.GroupNorm(num_groups=num_groups, num_channels=out_channels) \
                                # if normalization == "GN" else nn.BatchNorm3d(num_features=out_channels),
                                nn.InstanceNorm3d(out_channels),
                                nn.LeakyReLU(inplace=True))

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, normalization=normalization)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, normalization=normalization)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, normalization=normalization)
        self.b4 = _AsppPooling(in_channels, out_channels, normalization=normalization)

        self.project = nn.Sequential(nn.Conv3d(5 * out_channels, out_channels, 1, bias=False),
                                     # nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
                                     # if normalization == "GN" else nn.BatchNorm3d(num_features=out_channels),
                                     nn.InstanceNorm3d(out_channels),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Dropout(0.2))

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out