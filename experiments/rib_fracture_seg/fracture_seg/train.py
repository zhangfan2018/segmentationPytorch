
""" segmentation of rib fracture.
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
from data_processor.data_loader import CropDataSetLoader


# ---------------------------------------------Args config-------------------------------------------------- #
net_config = {"num_class": 1,
              "nb_filter": [8, 16, 32, 64, 128],
              "use_checkpoint": False}

args = ModelOptions("segmentation of rib").parse()
args.image_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/refine_data/crop_192_256_192_res0.7/image/"
args.mask_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/refine_data/crop_192_256_192_res0.7/mask/"
args.train_dataset = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/train_filename.csv"
args.val_dataset = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/val_filename.csv"
args.label = ["fracture"]
args.num_classes = 1
args.batch_size = 1
args.n_workers = 2
args.lr = 3e-4
args.epochs = 60
args.mode = "train"
args.out_dir = "./output/fracture_192-256-192_1216"


# --------------------------------------------Init--------------------------------------------------------- #
torch.cuda.manual_seed_all(args.seed) if args.cuda else torch.manual_seed(args.seed)
network = UNet(net_config)
train_dataLoader = CropDataSetLoader(csv_path=args.train_dataset, image_dir=args.image_dir,
                                     mask_dir=args.mask_dir, num_classes=args.num_classes, phase="train",
                                     window_level=[-1200, 1200])
val_dataLoader = CropDataSetLoader(csv_path=args.val_dataset, image_dir=args.image_dir,
                                   mask_dir=args.mask_dir, num_classes=args.num_classes, phase="val",
                                   window_level=[-1200, 1200])
model = SegmentationModel(args, network)


# --------------------------------------------Session------------------------------------------------------ #
if args.mode == "train":
    print('train mode')
    model.train(train_dataLoader, val_dataLoader)
elif args.mode == "val":
    print('validation mode')
    model.validate(val_dataLoader)