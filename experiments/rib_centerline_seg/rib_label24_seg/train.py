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
args.train_dataset = "/fileser/zhangfan/DataSet_1/rib_centerline_data/csv/rib_label_train.csv"
args.val_dataset = "/fileser/zhangfan/DataSet_1/rib_centerline_data/csv/rib_label_val.csv"
args.label = ["label_" + str(i) for i in range(1, 25)]
args.num_classes = 24
args.batch_size = 1
args.n_workers = 4
args.lr = 1e-3
args.epochs = 300
args.mode = "train"
args.out_dir = "./output/rib_label24_seg"


# --------------------------------------------Init--------------------------------------------------------- #
torch.cuda.manual_seed_all(args.seed) if args.cuda else torch.manual_seed(args.seed)
network = UNet(net_config)
train_dataLoader = DataSetLoader(csv_path=args.train_dataset, image_dir=args.image_dir,
                                 mask_dir=args.mask_dir, num_classes=args.num_classes, phase="train")
val_dataLoader = DataSetLoader(csv_path=args.val_dataset, image_dir=args.image_dir,
                               mask_dir=args.mask_dir, num_classes=args.num_classes, phase="val")
model = SegmentationModel(args, network)


# --------------------------------------------Session------------------------------------------------------ #
if args.mode == "train":
    print('train mode')
    model.train(train_dataLoader, val_dataLoader)
elif args.mode == "val":
    print('validation mode')
    model.validate(val_dataLoader)