"""Implementation of basic blocks of convolution neural network.
"""

import torch.nn as nn


def conv_kernel_3(in_channels, out_channels, model_type="2D", stride=1):
    """Convolution block of 3*3 or 3*3*3 kernel."""

    assert model_type == "2D" or model_type == "3D", "Don't exist {} type!".format(model_type)

    if model_type == "2D":
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    elif model_type == "3D":
        return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv_kernel_1(in_channels, out_channels, model_type="2D", stride=1):
    """Convolution block of 1*1 or 1*1*1 kernel."""

    assert model_type == "2D" or model_type == "3D", "Don't exist {} type!".format(model_type)

    if model_type == "2D":
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
    elif model_type == "3D":
        return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def batch_normal(channels, model_type="2D"):
    """Batch normalization"""

    assert model_type == "2D" or model_type == "3D", "Don't exist {} type!".format(model_type)

    if model_type == "2D":
        return nn.BatchNorm2d(channels)
    elif model_type == "3D":
        return nn.BatchNorm3d(channels)


def pool(kernel_size=2, stride=1, pool_type="max", model_type="2D"):
    """Pooling block of max or average pooling."""

    if pool_type == "max" and model_type == "2D":
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
    elif pool_type == "max" and model_type == "3D":
        return nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
    elif pool_type == "average" and model_type == "2D":
        return nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
    elif pool_type == "average" and model_type == "3D":
        return nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
    else:
        raise Exception("Don't exist {} type".format(pool_type))


def adaptive_pool(size=1, pool_type="max", model_type="2D"):
    """Global max or average pooling."""

    if pool_type == "max" and model_type == "2D":
        return nn.AdaptiveMaxPool2d(size)
    elif pool_type == "max" and model_type == "3D":
        return nn.AdaptiveMaxPool3d(size)
    elif pool_type == "average" and model_type == "2D":
        return nn.AdaptiveAvgPool2d(size)
    elif pool_type == "average" and model_type == "3D":
        return nn.AdaptiveAvgPool3d(size)
    else:
        raise Exception("Don't exist {} type".format(pool_type))


class residual_block(nn.Module):
    """Residual block."""

    def __init__(self, in_channels, out_channels, model_type="2D", stride=1):
        super(residual_block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv_kernel_3(in_channels, out_channels, model_type, stride)
        self.bn1 = batch_normal(out_channels, model_type)
        self.conv2 = conv_kernel_3(out_channels, out_channels, model_type)
        self.bn2 = batch_normal(out_channels, model_type)
        self.shortcut = conv_kernel_1(in_channels, out_channels, model_type, stride)
        self.bn3 = batch_normal(out_channels, model_type)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.bn3(self.shortcut(x))

        return self.relu(out)
