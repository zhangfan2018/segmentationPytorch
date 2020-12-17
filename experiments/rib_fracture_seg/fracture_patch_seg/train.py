
""" segmentation of rib fracture.
"""

import sys
import warnings
sys.path.append("/fileser/zhangfan/LungProject/lung_segment/")
warnings.filterwarnings("ignore")

import torch

from network.unet import UNet
from runner.runner import SegmentationModel
from runner.args import ModelOptions
from data_processor.data_loader import DataSetLoader


# ---------------------------------------------Args config-------------------------------------------------- #
net_config = {"num_class": 1,
              "nb_filter": [8, 16, 32, 64, 128],
              "use_checkpoint": False}

args = ModelOptions("segmentation of rib").parse()
args.image_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/crop_fracture_data/crop64_res0.75_patch/image/"
args.mask_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/crop_fracture_data/crop64_res0.75_patch/mask/"
args.train_dataset = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/train_patch_filename.csv"
args.val_dataset = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/val_patch_filename.csv"
args.label = ["fracture"]
args.num_classes = 1
args.batch_size = 45
args.n_workers = 4
args.lr = 1e-4
args.l2_penalty = 1e-6
args.epochs = 60
args.mode = "train"
args.out_dir = "./output/fracture_64-64-64_1216"


# --------------------------------------------Init--------------------------------------------------------- #
torch.cuda.manual_seed_all(args.seed) if args.cuda else torch.manual_seed(args.seed)
network = UNet(net_config)
train_dataLoader = DataSetLoader(csv_path=args.train_dataset, image_dir=args.image_dir,
                                 mask_dir=args.mask_dir, num_classes=args.num_classes, phase="train",
                                 window_level=[-1200, 1200])
val_dataLoader = DataSetLoader(csv_path=args.val_dataset, image_dir=args.image_dir,
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