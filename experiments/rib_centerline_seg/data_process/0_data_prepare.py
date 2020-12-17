
import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm

from utils.csv_tools import folder_to_csv, read_csv, csv_to_txt, write_csv, write_to_csv
from data_processor.data_process import ArgsConfig, PreProcessDataset, LoadData, \
    ProcessCropData, ProcessOriginalData, DataLoaderX
from data_processor.data_io import DataIO
from data_processor.data_prepare import divide_train_val_dataset

parser = argparse.ArgumentParser(description='Data preprocess of rib segmentation')
parser.add_argument('--dicom_to_nii', type=bool, default=False, help='convert dicom to nii.')
parser.add_argument('--folder_to_csv', type=bool, default=False, help='convert folder to csv file.')
parser.add_argument('--csv_to_txt', type=bool, default=False, help='convert csv to txt.')
parser.add_argument('--preprocess_data', type=bool, default=False, help='preprocess data.')
parser.add_argument('--divide_dataset', type=bool, default=False, help='divide dataset into train and validation set.')
parser.add_argument('--copy_dataset', type=bool, default=False,
                    help='copy dataset from source path to destination path.')
parser.add_argument('--merge_rib_centerline_mask', type=bool, default=False,
                    help='merge the mask of rib and centerline.')
parser.add_argument('--mha_to_nii', type=bool, default=True,
                    help='convert mha file to nii file.')
args = parser.parse_args()


# convert dicom to nii
if args.dicom_to_nii:
    dicom_dir = "/fileser/CT_RIB/data/dicom/"
    nii_dir = "/fileser/zhangfan/DataSet/lung_rib_data/test_data/image/"
    file_csv = "/fileser/zhangfan/DataSet/lung_rib_data/test_data/csv/bad_case_label.csv"
    data_loader = DataIO()
    file_names = read_csv(file_csv)[1:]
    for idx, file_name in enumerate(file_names):
        print("the processed number is {}/{}".format(idx, len(file_names)))
        uid = file_name[0] if type(file_name) == list else file_name
        dicom_path = dicom_dir + uid
        data_dict = data_loader.load_dicom_series(dicom_path)

        nii_path = nii_dir + uid + ".nii.gz"
        data_loader.save_medical_info_and_data(data_dict["image"], data_dict["origin"], data_dict["spacing"],
                                               data_dict["direction"], nii_path)


# folder to csv file
if args.folder_to_csv:
    data_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/mask/rib_centerline_mask/"
    csv_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/csv/rib_all_filename.csv"
    folder_to_csv(data_dir, ".nii.gz", csv_dir, header="seriesUid")


# csv to txt
if args.csv_to_txt:
    csv_path = "/fileser/zhangfan/DataSet/lung_rib_data/ori_dataset_csv/rib_data_list.csv"
    txt_path = "/fileser/zhangfan/DataSet/lung_rib_data/ori_dataset_csv/rib_data_list.txt"
    csv_to_txt(csv_path, txt_path)


# preprocess data
if args.preprocess_data:
    process_args = ArgsConfig().parse()
    process_args.csv_path = "/fileser/zhangfan/DataSet/pipeline_rib_mask/csv/rib_all_filename.csv"
    process_args.image_dir = "/fileser/CT_RIB/data/image/res0/"
    process_args.mask_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/mask/rib_centerline_mask/"
    process_args.out_image_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/image_refine/"
    process_args.out_mask_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/mask_refine/"
    process_args.is_smooth_mask = True
    process_args.is_label1_independent = True
    process_args.is_save_smooth_mask = False
    process_args.label = [1, 2]
    process_args.out_ori_size = [[128, 128, 128]]
    process_args.out_ori_spacing = []
    process_args.out_crop_size = [[224, 160, 224]]
    process_args.out_crop_spacing = []
    process_args.out_mask_stride = [1]

    dataset = PreProcessDataset(process_args)
    list_element = [LoadData(), ProcessOriginalData(), ProcessCropData()]
    dataset.set_pipeline(list_element)

    data_loader = DataLoaderX(
        dataset=dataset, batch_size=1, num_workers=12, shuffle=False)

    for index, flag in enumerate(data_loader):
        pass


# divide dataset into train and validation
if args.divide_dataset:
    dataset_csv_path = "/fileser/zhangfan/DataSet/pipeline_rib_mask/csv/rib_all_filename.csv"
    train_set_csv_path = "/fileser/zhangfan/DataSet/pipeline_rib_mask/csv/train_filename.csv"
    val_set_csv_path = "/fileser/zhangfan/DataSet/pipeline_rib_mask/csv/val_filename.csv"
    divide_train_val_dataset(dataset_csv_path, train_set_csv_path, val_set_csv_path, ratio=0.8)


# copying dataset from source path to destination path.
if args.copy_dataset:
    source_mask_dir = "/fileser/CT_RIB/data/mask_centerline/alpha_rib_centerline/"
    destination_mask_dir = "/fileser/zhangfan/DataSet/lung_rib_data/alpha_ribCenterlineSeg_twoStage/rib_mask_abnormal/"
    csv_path = "/fileser/zhangfan/DataSet/lung_rib_data/alpha_ribCenterlineSeg_twoStage/csv/rib_data_abnormal.csv"

    file_names = read_csv(csv_path)[1:]
    for file_name in file_names:
        ori_path = source_mask_dir + file_name[0] + ".nii.gz"
        dest_path = destination_mask_dir + file_name[0] + ".nii.gz"
        shutil.copy(ori_path, dest_path)


# merge the mask of rib and centerline.
if args.merge_rib_centerline_mask:
    csv_path = "/fileser/zhangfan/DataSet/lung_rib_data/csv/rib_data_20200813.csv"
    rib_dir = "/fileser/CT_RIB/data/mask/res0/"
    centerline_dir = "/fileser/CT_RIB/data/mask_centerline/alpha_rib_centerline/"
    res_dir = "/fileser/CT_RIB/data/mask_centerline/res0_old_pipeline/"

    files_list = read_csv(csv_path)[1:]
    files_list = [item[0] for item in files_list]

    data_loader = DataIO()
    for idx, file_name in enumerate(files_list):
        print("the processed number is {}/{}".format(idx, len(files_list)))
        rib_path = rib_dir + file_name + ".nii.gz"
        centerline_path = centerline_dir + file_name + ".nii.gz"
        res_path = res_dir + file_name + ".nii.gz"

        rib_data_dict = data_loader.load_nii_image(rib_path)
        centerline_data_dict = data_loader.load_nii_image(centerline_path)
        rid_mask = rib_data_dict["image"]
        centerline_mask = centerline_data_dict["image"]
        # centerline_mask[centerline_mask != 2] = 0

        res_mask = np.zeros_like(rid_mask, np.int8)
        res_mask[rid_mask == 1] = 1
        res_mask[centerline_mask == 2] = 2

        data_loader.save_medical_info_and_data(res_mask, rib_data_dict["origin"],
                                               rib_data_dict["spacing"],
                                               rib_data_dict["direction"],
                                               res_path)


# convert mha file to nii file.
if args.mha_to_nii:
    mha_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/alpha_413_centerline/ann_rib_mask_20201215/ori_mask_mha/"
    nii_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/alpha_413_centerline/ann_rib_mask_20201215/mask_nii/"
    csv_path = "/fileser/zhangfan/DataSet/pipeline_rib_mask/alpha_413_centerline/" \
               "ann_rib_mask_20201215/csv/rib_mask_20201215.csv"

    data_loader = DataIO()
    file_names = os.listdir(mha_dir)
    all_uids = []
    for file_name in tqdm(file_names):
        uid = file_name.split(".mha")[0]
        all_uids.append(uid)

        ori_path = mha_dir + file_name
        dst_path = nii_dir + uid + ".nii.gz"

        data_dict = data_loader.load_nii_image(ori_path)
        ori_mask = data_dict["image"]
        dst_mask = np.zeros(ori_mask.shape, np.uint8)
        dst_mask[dst_mask != 0] = 1

        data_loader.save_medical_info_and_data(dst_mask, data_dict["origin"], data_dict["spacing"],
                                               data_dict["direction"], dst_path)

    write_to_csv(all_uids, csv_path, header="seriesUid")