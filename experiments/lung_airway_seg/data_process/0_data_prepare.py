
import argparse

from utils.csv_tools import folder_to_csv
from data_processor.data_prepare import analysis_mask
from data_processor.data_prepare import divide_train_val_dataset
from data_processor.multi_thread_process import DataLoaderX, PreProcessDataset


parser = argparse.ArgumentParser(description='Data preprocess of lung segmentation')
parser.add_argument('--preprocess_data', type=bool, default=False, help='preprocess data.')
parser.add_argument('--folder_to_csv', type=bool, default=False, help='convert folder to csv file.')
parser.add_argument('--analysis_mask', type=bool, default=True, help='analysis mask information.')
parser.add_argument('--divide_dataset', type=bool, default=False, help='divide dataset into train and validation set.')
args = parser.parse_args()

if args.preprocess_data:
    """Data preprocess, including resample and crop mask to fixed size and spacing."""
    csv_path = "/fileser/zhangfan/DataSet/airway_segment_data/csv/luna_mask.csv"
    image_dir = "/fileser/DATA/IMAGE/LUNA/RAW_NII/"
    mask_dir = "/fileser/zhangfan/DataSet/airway_segment_data/luna_mask_nii/"
    image_out_dir = "/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/image_refine/"
    mask_out_dir = "/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/mask_refine/"

    dataset = PreProcessDataset(csv_path=csv_path, image_dir=image_dir, mask_dir=mask_dir, label=[3, 2],
                                is_smooth_mask=False, extend_size=20, cut_patch_mode="bbox",
                                is_label1_independent=False,
                                is_save_smooth_mask=False, is_save_crop_mask=False,
                                out_ori_size=[[128, 128, 128]], out_ori_spacing=[],
                                out_crop_size=[], out_crop_spacing=[], out_crop_patch_size=[],
                                out_image_dir=image_out_dir, out_mask_dir=mask_out_dir)

    train_loader = DataLoaderX(
        dataset=dataset,
        batch_size=1,
        num_workers=8,
        shuffle=False)

    for index, (flag, uid) in enumerate(train_loader):
        pass

if args.folder_to_csv:
    """Convert folder filename to .csv file."""
    data_dir = "/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/image_refine/ori_res1_1_1/"
    dataset_csv_path = "/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/csv/ori_res1_dataset.csv"
    folder_to_csv(file_dir=data_dir, data_suffix=".nii.gz", csv_path=dataset_csv_path, header=["seriesUid"])

if args.analysis_mask:
    """Analysis mask information."""
    csv_path = "/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/csv/ori_res1_dataset.csv"
    mask_dir = "/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/mask_refine/ori_res1_1_1/"
    analysis_mask(csv_path, mask_dir)

if args.divide_dataset:
    """Divide dataset into train and validation set"""
    dataset_csv_path = "/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/csv/ori_res1_dataset.csv"
    train_set_csv_path = "/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/csv/train_ori_res1.csv"
    val_set_csv_path = "/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/csv/val_ori_res1.csv"
    divide_train_val_dataset(dataset_csv_path, train_set_csv_path, val_set_csv_path)