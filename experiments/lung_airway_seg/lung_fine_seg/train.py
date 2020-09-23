""" segmentation of lung and airway in fine resolution.
"""

import os
import sys
import warnings
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("/fileser/zhangfan/LungProject/lung_segment/")
warnings.filterwarnings("ignore")

import torch

from network.unet import UNet
from runner.runner import SegmentationModel
from runner.args import ModelOptions
from data_processor.data_loader import DataSetLoader


# ---------------------------------------------Args config-------------------------------------------------- #
net_config = {"num_class": 2,
              "nb_filter": [8, 16, 32, 64, 128],
              "use_checkpoint": False}

args = ModelOptions("segmentation of lung").parse()
args.image_dir = "/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/image_refine/crop_256_192_256/"
args.mask_dir = "/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/mask_refine/crop_256_192_256/"
args.train_dataset = "/fileser/zhangfan/DataSet/airway_segment_data/csv/train_filename.csv"
args.val_dataset = "/fileser/zhangfan/DataSet/airway_segment_data/csv/val_filename.csv"
args.label = ["left_lung", "right_lung"]
args.num_classes = 2
args.batch_size = 1
args.n_workers = 4
args.lr = 1e-3
args.epochs = 150
args.mode = "train"
args.out_dir = "./output/lung_fine_seg"


# --------------------------------------------Init--------------------------------------------------------- #
torch.cuda.manual_seed_all(args.seed) if args.cuda else torch.manual_seed(args.seed)
network = UNet(net_config)
train_dataLoader = DataSetLoader(csv_path=args.train_dataset, image_dir=args.image_dir,
                                 mask_dir=args.mask_dir, num_classes=args.num_classes, phase="train",
                                 window_level=[-1200, 600])
val_dataLoader = DataSetLoader(csv_path=args.val_dataset, image_dir=args.image_dir,
                               mask_dir=args.mask_dir, num_classes=args.num_classes, phase="val",
                               window_level=[-1200, 600])
model = SegmentationModel(args, network)


# --------------------------------------------Session------------------------------------------------------ #
if args.mode == "train":
    print('train mode')
    model.train(train_dataLoader, val_dataLoader)
elif args.mode == "val":
    print('validation mode')
    model.validate(val_dataLoader)