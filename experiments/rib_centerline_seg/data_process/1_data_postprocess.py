
import os
import argparse
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from skimage.morphology import label

from data_processor.data_io import DataIO
from utils.csv_tools import read_csv, write_csv
from utils.mask_utils import find_rib_bound


parser = argparse.ArgumentParser(description='Data postprocess of rib segmentation')
parser.add_argument('--analysis_seg_result', type=bool, default=False, help='analysis segmentation result.')
parser.add_argument('--process_seg_result', type=bool, default=True, help='process segmentation result.')
args = parser.parse_args()


if args.analysis_seg_result:
    seg_mask_dir = "/fileser/zhangfan/DataSet/lung_rib_data/alpha_ribCenterlineSeg_twoStage/test_175/rib/"
    file_csv = "/fileser/zhangfan/DataSet/lung_rib_data/test_data/csv/seg_fps.csv"

    data_loader = DataIO()
    file_names = read_csv(file_csv)[1:]

    incomplete_case = {}
    axis_diff_median_z_all = []
    axis_diff_median_y_all = []
    axis_diff_median_x_all = []
    for idx, file_name in enumerate(file_names):
        print("the processed number is {}/{}".format(idx, len(file_names)))
        file_name = file_name[0] if type(file_name) == list else file_name
        mask_path = seg_mask_dir + file_name + ".nii.gz"

        mask_dict = data_loader.load_nii_image(mask_path)
        mask = mask_dict["image"]
        mask_size = mask.shape

        mask[mask != 0] = 1
        labeled_img, num = label(mask, neighbors=8, background=0, return_num=True)
        region_props = measure.regionprops(labeled_img)
        print("the number of rib is {}".format(num))

        # remove incomplete scanning case based on the number of rib label.
        if num < 18 or num > 48:
            incomplete_case[file_name] = num
            continue

        # extract rib centroid.
        x_centroid = {}
        y_centroid = {}
        z_centroid = {}

        for i in range(num):
            props = region_props[i]
            centroid = props.centroid
            x_centroid[i] = int(centroid[2])
            y_centroid[i] = int(centroid[1])
            z_centroid[i] = int(centroid[0])

        # filter FPs along the z axis.
        out_z_bound_idx, axis_diff_median_z = find_rib_bound(z_centroid, interval_value=20)
        out_y_bound_idx, axis_diff_median_y = find_rib_bound(y_centroid, interval_value=10)
        out_x_bound_idx, axis_diff_median_x = find_rib_bound(x_centroid)
        axis_diff_median_z_all.append(axis_diff_median_z)
        axis_diff_median_y_all.append(axis_diff_median_y)
        axis_diff_median_x_all.append(axis_diff_median_x)

        if len(out_z_bound_idx) or len(out_y_bound_idx):
            incomplete_case[file_name] = num

    for i in incomplete_case.keys():
        print("the uid:{}, the rib num:{}".format(i, incomplete_case[i]))

    plt.figure()
    plt.hist(axis_diff_median_z_all, bins=40)
    plt.title("the hist of z axis.")
    plt.show()

    plt.figure()
    plt.hist(axis_diff_median_y_all, bins=40)
    plt.title("the hist of y axis.")
    plt.show()

    plt.figure()
    plt.hist(axis_diff_median_x_all, bins=40)
    plt.title("the hist of x axis.")
    plt.show()


if args.process_seg_result:
    rib_dir = "/fileser/CT_RIB/data/mask_centerline/alpha_rib_centerline/"
    bad_case_csv = "/fileser/zhangfan/DataSet/lung_rib_data/csv/bad_case.csv"

    data_loader = DataIO()
    file_names = os.listdir(rib_dir)
    write_csv(bad_case_csv, ["seriesUid"], mul=False, mod="w")

    incomplete_case = {}
    refine_case = {}
    axis_diff_median_z_all = []
    axis_diff_median_y_all = []
    axis_diff_median_x_all = []
    for idx, file_name in enumerate(file_names):
        print("the processed number is {}/{}".format(idx, len(file_names)))
        file_name = file_name.split(".nii.gz")[0]
        mask_path = rib_dir + file_name + ".nii.gz"

        mask_dict = data_loader.load_nii_image(mask_path)
        mask_ori = mask_dict["image"]
        mask_size = mask_ori.shape

        mask = mask_ori.copy()
        mask[mask != 0] = 1
        labeled_img, num = label(mask, neighbors=8, background=0, return_num=True)
        region_props = measure.regionprops(labeled_img)
        print("the number of rib is {}".format(num))

        # remove incomplete scanning case based on the number of rib label.
        if num < 18 or num > 48:
            incomplete_case[file_name] = num
            write_csv(bad_case_csv, [file_name], mul=False, mod="a+")
            continue

        # extract rib centroid.
        x_centroid = {}
        y_centroid = {}
        z_centroid = {}

        for i in range(num):
            props = region_props[i]
            centroid = props.centroid
            x_centroid[i+1] = int(centroid[2])
            y_centroid[i+1] = int(centroid[1])
            z_centroid[i+1] = int(centroid[0])

        # filter FPs along the z axis.
        out_z_bound_idx, axis_diff_median_z = find_rib_bound(z_centroid, interval_value=20)
        out_y_bound_idx, axis_diff_median_y = find_rib_bound(y_centroid, interval_value=10)
        out_x_bound_idx, axis_diff_median_x = find_rib_bound(x_centroid)
        axis_diff_median_z_all.append(axis_diff_median_z)
        axis_diff_median_y_all.append(axis_diff_median_y)
        axis_diff_median_x_all.append(axis_diff_median_x)

        # remove incomplete scanning case based on the number of rib label.
        new_num = num - len(out_z_bound_idx) - len(out_y_bound_idx)
        if new_num < 18:
            incomplete_case[file_name] = new_num
            write_csv(bad_case_csv, [file_name], mul=False, mod="a+")

        for i in out_z_bound_idx:
            labeled_img[labeled_img == i] = 0
        for i in out_y_bound_idx:
            labeled_img[labeled_img == i] = 0
        if len(out_z_bound_idx) or len(out_y_bound_idx):
            refine_case[file_name] = num
            labeled_img[labeled_img != 0] = 1
            labeled_img[mask_ori == 2] = 2
            data_loader.save_medical_info_and_data(labeled_img, mask_dict["origin"],
                                                   mask_dict["spacing"], mask_dict["direction"], mask_path)

    for i in incomplete_case.keys():
        print("the uid:{}, the rib num:{}".format(i, incomplete_case[i]))

    for i in refine_case.keys():
        print("the refine case uid:{}".format(i))

    plt.figure()
    plt.hist(axis_diff_median_z_all, bins=40)
    plt.title("the hist of z axis.")
    plt.show()

    plt.figure()
    plt.hist(axis_diff_median_y_all, bins=40)
    plt.title("the hist of y axis.")
    plt.show()

    plt.figure()
    plt.hist(axis_diff_median_x_all, bins=40)
    plt.title("the hist of x axis.")
    plt.show()






