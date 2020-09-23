
""" segmentation of rib and centerline in fine resolution.
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

args = ModelOptions("segmentation of rib and centerline").parse()
args.image_dir = "/fileser/CT_RIB/data/image_refine/crop_224_160_224/"
args.mask_dir = "/fileser/CT_RIB/data/mask_refine/crop_224_160_224/"
args.train_dataset = "/fileser/zhangfan/DataSet/lung_rib_data/csv/train_filename_all.csv"
args.val_dataset = "/fileser/zhangfan/DataSet/lung_rib_data/csv/val_filename_all.csv"
args.label = ["rib", "centerline"]
args.num_classes = 2
args.batch_size = 1
args.n_workers = 4
args.lr = 1e-3
args.epochs = 300
args.mode = "train"
args.out_dir = "./output/rib_centerline_fine_seg"
exclude_csv = "/fileser/zhangfan/DataSet/lung_rib_data/csv/bad_case.csv"


# --------------------------------------------Init--------------------------------------------------------- #
torch.cuda.manual_seed_all(args.seed) if args.cuda else torch.manual_seed(args.seed)
network = UNet(net_config)
train_dataLoader = DataSetLoader(csv_path=args.train_dataset, image_dir=args.image_dir,
                                 mask_dir=args.mask_dir, num_classes=args.num_classes, phase="train",
                                 file_exclude_csv=exclude_csv, window_level=[-1200, 1200])
val_dataLoader = DataSetLoader(csv_path=args.val_dataset, image_dir=args.image_dir,
                               mask_dir=args.mask_dir, num_classes=args.num_classes, phase="val",
                               file_exclude_csv=exclude_csv, window_level=[-1200, 1200])
model = SegmentationModel(args, network)


# --------------------------------------------Session------------------------------------------------------ #
if args.mode == "train":
    print('train mode')
    model.train(train_dataLoader, val_dataLoader)
elif args.mode == "val":
    print('validation mode')
    model.validate(val_dataLoader)