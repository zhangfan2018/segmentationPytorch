"""Convert torch into torch script model which run on libtorch library.
"""

import os
import torch

from network.unet import UNet


def load_model_weights(model, model_dir):
    if os.path.exists(model_dir):
        print('Loading model weights...')
        checkpoint = torch.load(model_dir)
        model = model.load_state_dict({k.replace('module.', ""): v for k, v in checkpoint['state_dict'].items()})
    return model


def set_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def convert_model_to_torchscript(model, dummy_input, save_path):
    set_requires_grad(model)
    model.eval()
    traced_script_module = torch.jit.trace(model, dummy_input)
    print(traced_script_module.code)
    traced_script_module.save(save_path)


if __name__ == "__main__":
    model_dir = "model.pt"
    save_dir = "torch_script.pt"
    model = UNet()
    model = load_model_weights(model, model_dir)
    model = model.cuda()
    model.eval()
    dummy_input = torch.rand(1, 1, 128, 128, 128).float().cuda()
    convert_model_to_torchscript(model, dummy_input, save_dir)

