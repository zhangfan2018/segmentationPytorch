"""Calculate the parameters and flops of model.
"""

import torch
import warnings
from thop import profile

from network.unet import UNet

warnings.filterwarnings("ignore")


def get_layer_param(net):

    return sum([torch.numel(param) for param in net.parameters()])


def get_params_flops(net, input_array):
    flops, params = profile(net, inputs=(input_array))

    return params, flops


if __name__ == '__main__':
    input_image = torch.randn(1, 1, 128, 128, 128)
    model = UNet()
    flops, params = profile(model, inputs=(input_image,))
    print("The number of parameters of model is {} MB".format(get_layer_param(model) / (1024*1024)))
    print("The flops is {} GB".format(flops / (1024 * 1024 * 1024)))
    print("The params is {} MB".format(params / ( 1024 * 1024)))



