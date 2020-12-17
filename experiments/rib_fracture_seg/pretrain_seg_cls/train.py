
""" segmentation of rib fracture.
"""

import os
import sys
import warnings
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("/fileser/zhangfan/LungProject/lung_segment/")
warnings.filterwarnings("ignore")

import torch

from network.unet_multi_task import UNetMultiTask
from runner.multi_task_runner import SegmentationModel
from runner.args import ModelOptions
from data_processor.data_loader import LabeledDataSetLoader


# ---------------------------------------------Args config-------------------------------------------------- #
net_config = {"num_class": 1,
              "nb_filter": [8, 16, 32, 64, 128],
              "use_checkpoint": False}

args = ModelOptions("segmentation of rib").parse()
args.image_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/crop_fracture_data/crop64_res0.75_patch/image/"
args.mask_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/crop_fracture_data/crop64_res0.75_patch/mask/"
args.train_dataset = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/train_crop64_res0.75.csv"
args.val_dataset = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/val_crop64_res0.75.csv"
args.label = ["fracture"]
args.num_classes = 1
args.batch_size = 32
args.n_workers = 4
args.lr = 1e-4
args.l2_penalty = 0
args.epochs = 60
args.mode = "train"
args.out_dir = "./output/fracture_patch_1217"
args.task = "multi"
args.is_distributed_train = False
# args.weight_dir = "/fileser/zhangfan/LungProject/lung_segment/experiments/rib_centerline_seg/" \
#                   "rib_centerline_fine_seg/output/rib_fine_seg_1029/models_2020-10-29_19-13-03/models/model_149.pt"


# --------------------------------------------Init--------------------------------------------------------- #
torch.cuda.manual_seed_all(args.seed) if args.cuda else torch.manual_seed(args.seed)
network = UNetMultiTask(net_config)
train_dataLoader = LabeledDataSetLoader(csv_path=args.train_dataset, image_dir=args.image_dir,
                                        mask_dir=args.mask_dir, num_classes=args.num_classes, phase="train",
                                        is_sample=True, window_level=[-1200, 1200])
val_dataLoader = LabeledDataSetLoader(csv_path=args.val_dataset, image_dir=args.image_dir,
                                      mask_dir=args.mask_dir, num_classes=args.num_classes, phase="val",
                                      is_sample=False, window_level=[-1200, 1200])
model = SegmentationModel(args, network)


# --------------------------------------------Session------------------------------------------------------ #
if args.mode == "train":
    print('train mode')
    model.train(train_dataLoader, val_dataLoader)
elif args.mode == "val":
    print('validation mode')
    model.validate(val_dataLoader)