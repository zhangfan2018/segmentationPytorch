""" segmentation of fracture lesion.
"""

import sys
import warnings
sys.path.append("/fileser/zhangfan/LungProject/lung_segment/")
warnings.filterwarnings("ignore")

import torch

from network.unet_deploy import UNet
from runner.runner import SegmentationModel
from runner.args import ModelOptions
from data_processor.data_loader import FracturePatchDataset


# ---------------------------------------------Args config-------------------------------------------------- #
net_config = {"num_class": 1,
              "nb_filter": [8, 16, 32, 64, 128],
              "use_checkpoint": False}

args = ModelOptions("segmentation of rib fracture").parse()
args.image_dir = "/fileser/CT_RIB/data/image/res0/"
args.mask_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/test_mask/rib/"
args.val_dataset = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/test_148.csv"
args.weight_dir = "/fileser/zhangfan/LungProject/lung_segment/experiments/rib_fracture_seg/fracture_patch_seg/output/" \
                  "fracture_64-64-64_1212/models_2020-12-11_19-02-14/models/model_112.pt"
args.label = ["fracture"]
args.num_classes = 1
args.mode = "test"
args.out_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/test_result/fracture_patch_1216_1"


# --------------------------------------------Init--------------------------------------------------------- #
torch.cuda.manual_seed_all(args.seed) if args.cuda else torch.manual_seed(args.seed)
network = UNet(net_config)
test_dataLoader = FracturePatchDataset(csv_path=args.val_dataset, image_dir=args.image_dir,
                                       centerline_dir="/fileser/zhangfan/DataSet/pipeline_rib_mask/"
                                                      "alpha_413_centerline/num148_det_thres0.3_ori_result/",
                                       bbox_radius=32, stride=32)
model = SegmentationModel(args, network)


# --------------------------------------------Session------------------------------------------------------ #
print('test mode')
model.test_frac_patch_seg(test_dataLoader)
