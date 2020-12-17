
import os
import random
import argparse
import numpy as np

from utils.csv_tools import read_txt, read_csv, write_to_csv


parser = argparse.ArgumentParser(description='Data preprocess of fracture segmentation')
parser.add_argument('--divide_dataset', type=bool, default=False, help='divide dataset into train and validation set.')
parser.add_argument('--folder_to_csv', type=bool, default=True, help='convert folder to csv file.')
args = parser.parse_args()


if args.divide_dataset:
    uid_list_txt = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/rib_train_uids_1592.txt"
    train_csv_path = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/train_filename.csv"
    val_csv_path = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/val_filename.csv"

    ratio = 0.8
    data_names = read_txt(uid_list_txt)
    random.shuffle(data_names)
    train_contents = data_names[:int(len(data_names)*ratio)]
    val_contents = data_names[int(len(data_names)*ratio):]
    write_to_csv(train_contents, train_csv_path, header="seriesUid")
    write_to_csv(val_contents, val_csv_path, header="seriesUid")


if args.folder_to_csv:
    train_uid_csv = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/train_filename.csv"
    val_uid_csv = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/val_filename.csv"
    folder_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/crop_fracture_data/crop64_res0.75_patch/mask/"

    out_train_csv = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/train_patch_filename.csv"
    out_val_csv = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/val_patch_filename.csv"

    train_uids = read_csv(train_uid_csv)[1:]
    train_uids = [item[0] for item in train_uids]

    val_uids = read_csv(val_uid_csv)[1:]
    val_uids = [item[0] for item in val_uids]

    all_file_names = os.listdir(folder_dir)

    all_train_file_names = []
    all_val_file_names = []

    for file_name in all_file_names:
        uid = file_name.split("_")[0]
        patch_uid = file_name.split(".nii.gz")[0]
        candidate_type = file_name.split("_")[1]
        is_positive = candidate_type.split("-")[1]
        is_keep = True
        if is_positive != "1":
            is_keep = False
            # is_sample = np.random.randint(2)
            # is_keep = False if is_sample == 0 else True
        if not is_keep:
            continue
        if uid in train_uids:
            all_train_file_names.append(patch_uid)
        elif uid in val_uids:
            all_val_file_names.append(patch_uid)

    write_to_csv(all_train_file_names, out_train_csv, "fileName")
    write_to_csv(all_val_file_names, out_val_csv, "fileName")
