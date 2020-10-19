
""" Convert torch model in pytorch to torchscript model in libtorch.
"""
import os
import sys
import torch
sys.path.append("/fileser/zhangfan/LungProject/lung_segment/")

from network.deploy_unet_block3 import UNet
from model_utils.convert_model_to_torchscript import load_model_weights, convert_model_to_torchscript


if __name__ == "__main__":
    model_dir = "/fileser/zhangfan/LungProject/lung_segment/experiments/rib_centerline_seg/rib_centerline_fine_seg/" \
                "output/rib_centerline_fine_seg_block3_num158/models_2020-10-13_14-31-40/models/model_486.pt"
    save_dir = "/fileser/zhangfan/models_zoo/fine_rib_seg_tmp/model.pt"

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    net_config = {"num_class": 2,
                  "nb_filter": [12, 24, 48, 96],
                  "input_size": [224, 160, 224],
                  "clip_window": [-1200, 1200]}
    model = UNet(net_config)
    model = load_model_weights(model, model_dir)
    model = model.cuda().half()
    dummy_input = torch.rand(1, 1, 512, 512, 512).float().cuda().half()
    convert_model_to_torchscript(model, dummy_input, save_dir)
