
import torch
from network.unet_multi_task_IN import UNetMultiTask

weight_dir = "/fileser/zhangfan/LungProject/lung_segment/experiments/rib_fracture_seg/pretrain_seg_cls/" \
                  "output/fracture_patch_1215/models_2020-12-15_16-04-00/models/model_019.pt"

checkpoint = torch.load(weight_dir)
pretrained_params = {k.replace('module.', ""): v for k, v in checkpoint['state_dict'].items()}

network = UNetMultiTask()
net_state_dict = network.state_dict()

pretrained_dict = {k: v for k, v in pretrained_params.items() if k in net_state_dict}

for k, v in net_state_dict.items():
    print(k)
