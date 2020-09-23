""" segmentation of lung and airway in coarse resolution.
"""

import os
import sys
import warnings
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("/fileser/zhangfan/LungProject/lung_segment/")
warnings.filterwarnings("ignore")

import torch

from network.unet_deploy import UNet
from runner.runner import SegmentationModel
from runner.args import ModelOptions
from data_processor.data_loader import DataSetLoader


# ---------------------------------------------Args config-------------------------------------------------- #
net_config = {"num_class": 3,
              "nb_filter": [8, 16, 32, 64, 128],
              "use_checkpoint": False}

args = ModelOptions("segmentation of lung and airway").parse()
args.image_dir = "/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/image_refine/ori_128_128_128/"
args.mask_dir = "/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/mask_refine/ori_128_128_128/"
args.val_dataset = "/fileser/zhangfan/DataSet/airway_segment_data/csv/val_filename.csv"
args.weight_dir = "/fileser/zhangfan/LungProject/lung_segment/experiments/lung_airway_seg/output/" \
                  "lung_airway_coarse_seg/models_2020-09-10_09-46-22/models/model_000.pt"
args.label = ["left_lung", "right_lung", "airway"]
args.num_classes = 3
args.batch_size = 1
args.n_workers = 2
args.mode = "test"
args.is_save_script_model = False
args.out_dir = "./output/test_lung_airway_coarse_seg"


# --------------------------------------------Init--------------------------------------------------------- #
torch.cuda.manual_seed_all(args.seed) if args.cuda else torch.manual_seed(args.seed)
network = UNet(net_config)
val_dataLoader = DataSetLoader(csv_path=args.val_dataset, image_dir=args.image_dir,
                               mask_dir=args.mask_dir, num_classes=args.num_classes, phase="test",
                               window_level=[-1200, 600])
model = SegmentationModel(args, network)


# --------------------------------------------Session------------------------------------------------------ #
print('test mode')
model.test(val_dataLoader)