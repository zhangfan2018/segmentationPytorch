""" Convert pytorch model to torchscript model.
"""
import sys
sys.path.append("/fileser/zhangfan/LungProject/lung_segment/")
import os
import torch

from network.deploy_unet_block3 import UNet
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

pytorch_model_dir = "/fileser/zhangfan/LungProject/lung_segment/experiments/rib_centerline_seg/rib_centerline_fine_seg/" \
                    "output/rib_centerline_fine_seg_block3_num158/models_2020-10-13_14-31-40/models/model_486.pt"
torchscript_model_dir = "/fileser/zhangfan/models_zoo/fine_rib_seg_tmp/model.pt"

model = UNet()
if os.path.exists(pytorch_model_dir):
    print('Loading pre_trained model...')
    checkpoint = torch.load(pytorch_model_dir)
    model.load_state_dict({k.replace('module.', ""): v for k, v in checkpoint['state_dict'].items()})

# model = model.cuda().half()
model = model.cuda()
for param in model.parameters():
    param.requires_grad = False

# convert torch script model.
model.eval()
# dummy_input = torch.rand(1, 1, 256, 192, 256).float().cuda().half()
dummy_input = torch.rand(1, 1, 256, 192, 256).float().cuda()
with torch.no_grad():
    traced_script_module = torch.jit.trace(model, dummy_input)
print(traced_script_module.code)
traced_script_module.save(torchscript_model_dir)