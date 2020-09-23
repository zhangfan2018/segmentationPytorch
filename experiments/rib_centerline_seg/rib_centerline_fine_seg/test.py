""" segmentation of rib and centerline in fine resolution.
"""

import os
import sys
import warnings
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("/fileser/zhangfan/LungProject/lung_segment/")
warnings.filterwarnings("ignore")

import torch

from network.unet_preprocess_deploy import UNet
from runner.runner import SegmentationModel
from runner.args import ModelOptions
from data_processor.data_loader import DataSetLoader


# ---------------------------------------------Args config-------------------------------------------------- #
net_config = {"num_class": 2,
              "nb_filter": [8, 16, 32, 64, 128],
              "input_size": [224, 160, 224],
              "clip_window": [-1200, 1200]}

args = ModelOptions("segmentation of rib").parse()
args.image_dir = "/fileser/CT_RIB/data/image_refine/crop_res1_1_1/"
args.mask_dir = "/fileser/CT_RIB/data/mask_refine/crop_res1_1_1/"
args.val_dataset = "/fileser/zhangfan/DataSet/lung_rib_data/csv/val_filename.csv"
args.weight_dir = "/fileser/zhangfan/LungProject/lung_segment/experiments/rib_centerline_seg/rib_centerline_fine_seg" \
                  "/output/rib_centerline_fine_seg/models_2020-09-21_18-36-25/models/model_300.pt"
args.label = ["rib", "centerline"]
args.num_classes = 2
args.batch_size = 1
args.n_workers = 2
args.mode = "test"
args.is_save_script_model = True
args.is_fp16 = True
args.is_save_mask = False
args.is_upsample_mask = False
args.out_dir = "./output/test_rib_fine"
args.gpu_ids = "4"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
exclude_csv = "/fileser/zhangfan/DataSet/lung_rib_data/csv/bad_case.csv"


# --------------------------------------------Init--------------------------------------------------------- #
torch.cuda.manual_seed_all(args.seed) if args.cuda else torch.manual_seed(args.seed)
network = UNet(net_config)
val_dataLoader = DataSetLoader(csv_path=args.val_dataset, image_dir=args.image_dir,
                               mask_dir=args.mask_dir, num_classes=args.num_classes, phase="test",
                               file_exclude_csv=exclude_csv, window_level=[-1200, 1200])
model = SegmentationModel(args, network)


# --------------------------------------------Session------------------------------------------------------ #
print('test mode')
model.test(val_dataLoader)