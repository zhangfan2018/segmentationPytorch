""" segmentation of rib label in fine resolution.
"""

import os
import sys
import warnings
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("/fileser/zhangfan/LungProject/lung_segment/")
warnings.filterwarnings("ignore")

import torch

from network.unet_downsample2X import UNet
from runner.runner import SegmentationModel
from runner.args import ModelOptions
from data_processor.data_loader import DataSetLoader


# ---------------------------------------------Args config-------------------------------------------------- #
net_config = {"num_class": 24,
              "nb_filter": [12, 24, 48, 96, 192],
              "use_checkpoint": False}

args = ModelOptions("segmentation of rib label24").parse()
args.image_dir = "/fileser/CT_RIB/data/rib_image_refine/crop_256_192_256/"
args.mask_dir = "/fileser/CT_RIB/data/rib_mask_refine/crop_256_192_256/"
args.val_dataset = "/fileser/zhangfan/DataSet_1/rib_centerline_data/csv/rib_label_val.csv"
args.weight_dir = "/fileser/zhangfan/LungProject/lung_segment/experiments/rib_centerline_seg/output/rib_label24_seg/" \
                  "models_2020-09-09_21-25-57/models/model_109.pt"
args.label = ["label_" + str(i) for i in range(1, 25)]
args.num_classes = 24
args.batch_size = 1
args.n_workers = 2
args.mode = "test"
args.is_save_script_model = False
args.is_save_mask = True
args.is_upsample_mask = True
args.out_dir = "./output/test_rib_label24"
args.gpu_ids = "4"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids


# --------------------------------------------Init--------------------------------------------------------- #
torch.cuda.manual_seed_all(args.seed) if args.cuda else torch.manual_seed(args.seed)
network = UNet(net_config)
val_dataLoader = DataSetLoader(csv_path=args.val_dataset, image_dir=args.image_dir,
                               mask_dir=args.mask_dir, num_classes=args.num_classes, phase="test")
model = SegmentationModel(args, network)


# --------------------------------------------Session------------------------------------------------------ #
print('test mode')
model.test(val_dataLoader)