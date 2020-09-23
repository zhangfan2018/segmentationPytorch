
import shutil
import argparse

from utils.csv_tools import folder_to_csv, read_csv
from data_processor.data_process import ArgsConfig, PreProcessDataset, LoadData, \
    ProcessCropData, ProcessOriginalData, DataLoaderX
from data_processor.data_io import DataIO
from data_processor.data_prepare import divide_train_val_dataset

parser = argparse.ArgumentParser(description='Data preprocess of rib segmentation')
parser.add_argument('--dicom_to_nii', type=bool, default=False, help='convert dicom to nii.')
parser.add_argument('--folder_to_csv', type=bool, default=False, help='convert folder to csv file.')
parser.add_argument('--preprocess_data', type=bool, default=True, help='preprocess data.')
parser.add_argument('--divide_dataset', type=bool, default=False, help='divide dataset into train and validation set.')
parser.add_argument('--copy_dataset', type=bool, default=False, help='copy dataset from source path to destination path.')
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
    data_dir = "/fileser/CT_RIB/data/mask/rib_label_res0/"
    csv_dir = "/fileser/zhangfan/DataSet_1/rib_centerline_data/csv/rib_label_all.csv"
    folder_to_csv(data_dir, ".nii.gz", csv_dir, header="seriesUid")


# preprocess data
if args.preprocess_data:
    process_args = ArgsConfig().parse()
    process_args.csv_path = "/fileser/zhangfan/DataSet/lung_rib_data/csv/rib_data_20200813.csv"
    process_args.image_dir = "/fileser/CT_RIB/data/image/res0/"
    process_args.mask_dir = "/fileser/CT_RIB/data/mask_centerline/alpha_rib_centerline/"
    process_args.out_image_dir = "/fileser/CT_RIB/data/image_refine/"
    process_args.out_mask_dir = "/fileser/CT_RIB/data/mask_refine/"
    process_args.is_smooth_mask = True
    process_args.is_label1_independent = True
    process_args.is_save_smooth_mask = False
    process_args.label = [2, 2]
    process_args.out_ori_size = []
    process_args.out_ori_spacing = []
    process_args.out_crop_size = [[224, 160, 224]]
    process_args.out_crop_spacing = [[1, 1, 1]]
    process_args.out_mask_stride = [1]

    dataset = PreProcessDataset(process_args)
    list_element = [LoadData(), ProcessOriginalData(), ProcessCropData()]
    dataset.set_pipeline(list_element)

    data_loader = DataLoaderX(
        dataset=dataset, batch_size=1, num_workers=8, shuffle=False)

    for index, flag in enumerate(data_loader):
        pass


# divide dataset into train and validation
if args.divide_dataset:
    dataset_csv_path = "/fileser/zhangfan/DataSet_1/rib_centerline_data/csv/rib_label_all.csv"
    train_set_csv_path = "/fileser/zhangfan/DataSet_1/rib_centerline_data/csv/rib_label_train.csv"
    val_set_csv_path = "/fileser/zhangfan/DataSet_1/rib_centerline_data/csv/rib_label_val.csv"
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