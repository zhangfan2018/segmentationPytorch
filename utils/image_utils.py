
"""image preprocess tools
method:
clip_image
normalize_hu
normalize_mean_std
clip_and_normalize
crop_image_mask
padding_image
"""

import random
import numpy as np


def clip_image(image, min_window=-1200, max_window=600):
    """
    Clip image in a range of [min_window, max_window] in HU values.
    """
    image = np.clip(image, min_window, max_window)

    return image


def normalize_hu(image, min_window=-1200.0, max_window=600.0):
    """
    Normalize image HU value to [-1, 1] using window of
    [min_window, max_window].
    """
    image = (image - min_window) / (max_window - min_window)
    image = image * 2 - 1.0
    image = image.clip(-1, 1)

    return image


def normalize_mean_std(image, global_mean=None, global_std=None):
    """
    Normalize image by (Voxel - mean) / std, the operate should
    be local or global normalization.
    """
    if not global_mean or not global_std:
        mean = np.mean(image)
        std = np.std(image)
    else:
        mean, std = global_mean, global_std
    image = (image - mean) / (std + 1e-5)

    return image


def clip_and_normalize(image, min_window=-1200, max_window=600):
    """
    Clip image in a range of [min_window, max_window] in HU values.
    """
    image = np.clip(image, min_window, max_window)
    mean = np.mean(image)
    std = np.std(image)
    image = (image - mean) / (std + 1e-5)

    return image


def crop_image_mask(image, mask, margin=[20, 20, 20]):
    """crop image and mask based on mask bbox"""
    """
    Args:
        image(numpy array): image array.
        mask(numpy array): mask array.
        margin(list): extend margin
    Return:
        image_out: cropped image.
        mask_out: cropped mask.
    """
    t_mask = mask > 0
    zz, yy, xx = np.where(t_mask)

    bbox = np.array([[np.min(zz), np.max(zz)], [np.min(yy), np.max(yy)],
                    [np.min(xx), np.max(xx)]])

    extend_bbox = np.concatenate(
        [np.max([[0, 0, 0], bbox[:, 0] - margin], axis=0)[:, np.newaxis],
         np.min([image.shape, bbox[:, 1] + margin], axis=0)[:, np.newaxis]], axis=1)

    crop_image = image[extend_bbox[0, 0]:extend_bbox[0, 1],
                       extend_bbox[1, 0]: extend_bbox[1, 1],
                       extend_bbox[2, 0]: extend_bbox[2, 1]]

    crop_mask = mask[extend_bbox[0, 0]:extend_bbox[0, 1],
                     extend_bbox[1, 0]: extend_bbox[1, 1],
                     extend_bbox[2, 0]: extend_bbox[2, 1]]

    image_out = crop_image.copy()
    mask_out = crop_mask.copy()

    return image_out, mask_out


def padding_image(image, size=[64, 64, 64]):
    """padding image to fixed size"""
    """
    Args:
        image(numpy array): image array.
        size(list): padding image size.
    Return:
        pad_image(numpy array): padding image array.
    """
    row, column, slices = image.shape[0], image.shape[1], image.shape[2]
    radius = [row//2, column//2, slices//2]
    pad_image = np.zeros(size)
    x_begin = size[0] // 2 - radius[0]
    x_end = x_begin + row
    y_begin = size[1] // 2 - radius[1]
    y_end = y_begin + column
    z_begin = size[2] // 2 - radius[2]
    z_end = z_begin + slices
    pad_image[x_begin:x_end, y_begin:y_end, z_begin:z_end] = image

    return pad_image