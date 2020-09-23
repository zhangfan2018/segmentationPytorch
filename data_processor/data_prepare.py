""" Preprocess image dataset.
method:
divide_train_val_dataset
analysis_mask
copy_image_data
"""

import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt

from data_processor.data_io import DataIO
from utils.csv_tools import read_csv, write_to_csv


def divide_train_val_dataset(csv_dir, train_dir, val_dir, ratio=0.8):
    """divide dataset into train and val set."""
    data_names = read_csv(csv_dir)[1:]
    random.shuffle(data_names)
    train_contents = data_names[:int(len(data_names)*ratio)]
    val_contents = data_names[int(len(data_names)*ratio):]
    write_to_csv(train_contents, train_dir, header="seriesUid")
    write_to_csv(val_contents, val_dir, header="seriesUid")


def analysis_mask(csv_path, mask_dir):
    """analysis the scale and spacing of mask data."""
    data_loader = DataIO()
    file_names = read_csv(csv_path)[1:]

    mask_size = []
    mask_spacing = []
    roi_bbox = []
    roi_size = []
    for idx, file_name in enumerate(file_names):
        print("the processed number is {}/{}".format(idx, len(file_names)))
        file_name = file_name[0] if type(file_name) == list else file_name
        file_path = os.path.join(mask_dir, file_name) + ".nii.gz"
        if os.path.exists(file_path):
            data_dict = data_loader.load_nii_image(file_path)
            mask = data_dict["image"]
            spacing_ori = data_dict["spacing"]
            spacing = [spacing_ori[2], spacing_ori[1], spacing_ori[0]]
            mask_size.append(mask.shape)
            mask_spacing.append(spacing)

            t_mask = mask > 0
            zz, yy, xx = np.where(t_mask)
            bbox = [np.min(zz), np.min(yy), np.min(xx),  np.max(zz), np.max(yy), np.max(xx)]
            scale = [bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]]
            roi_bbox.append(bbox)
            roi_size.append(scale)

    mask_size = np.array(mask_size)
    mask_spacing = np.array(mask_spacing)
    roi_bbox = np.array(roi_bbox)
    roi_size = np.array(roi_size)

    # plot the hist of the size of original and roi image
    plt.figure()
    plt.subplot(321)
    plt.hist(mask_size[:, 0], bins=40)
    plt.title("the original mask size z")
    plt.subplot(322)
    plt.hist(roi_size[:, 0], bins=40)
    plt.title("the roi mask size z")
    plt.subplot(323)
    plt.hist(mask_size[:, 1], bins=40)
    plt.title("the original mask size y")
    plt.subplot(324)
    plt.hist(roi_size[:, 1], bins=40)
    plt.title("the roi mask size y")
    plt.subplot(325)
    plt.hist(mask_size[:, 2], bins=40)
    plt.title("the original mask size x")
    plt.subplot(326)
    plt.hist(roi_size[:, 2], bins=40)
    plt.title("the roi mask size x")
    plt.show()

    # calculate the mean and std of the size of original image.
    print("the mean spacing of original mask is z:{}, y:{}, x:{}".format(np.mean(mask_spacing[:, 0]),
                                                                         np.mean(mask_spacing[:, 1]),
                                                                         np.mean(mask_spacing[:, 2])))

    print("the mean shape of original mask is z:{}, y:{}, x:{}".format(np.mean(mask_size[:, 0]),
                                                                       np.mean(mask_size[:, 1]),
                                                                       np.mean(mask_size[:, 2])))

    print("the max shape of original mask is z:{}, y:{}, x:{}".format(np.max(mask_size[:, 0]),
                                                                      np.max(mask_size[:, 1]),
                                                                      np.max(mask_size[:, 2])))

    # calculate the min, max and mean of the size of roi image.
    print("the mean shape of roi mask is x:{}, y:{}, z:{}".format(np.mean(roi_size[:, 0]),
                                                                  np.mean(roi_size[:, 1]),
                                                                  np.mean(roi_size[:, 2])))
    print("the max shape of roi mask is x:{}, y:{}, z:{}".format(np.max(roi_size[:, 0]),
                                                                 np.max(roi_size[:, 1]),
                                                                 np.max(roi_size[:, 2])))


def copy_image_data(ori_dir, dest_dir):
    file_names = os.listdir(ori_dir)
    for idx, file_name in enumerate(file_names):
        print("the processed number is {}/{}".format(idx, len(file_names)))
        ori_path = os.path.join(ori_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        shutil.copy(ori_path, dest_path)