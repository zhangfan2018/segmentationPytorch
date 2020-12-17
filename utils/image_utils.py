
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


def trans_physical_2_index(spacing, origin, physical_point):
    """
    Convert physical coords to voxel coords.
    """
    index_point = []
    for i in range(0, 3):
        index_point.append((physical_point[i] - origin[i]) / spacing[i])
    return index_point


def trans_index_2_physical(spacing, origin, index_point):
    """
    Convert voxel coords to physical coords.
    """
    physical_point = []
    for i in range(0, 3):
        physical_point.append(index_point[i] * spacing[i] + origin[i])
    return physical_point


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


def crop_image_mask_by_multi_info(image, mask, rib, x_bbox, z_num=2, margin=10):
    t_mask = rib > 0
    slices, _, _ = image.shape
    zz, yy, xx = np.where(t_mask)

    bbox = np.array([[np.min(zz), np.max(zz)], [np.min(yy), np.max(yy)],
                    [np.min(xx), np.max(xx)]])

    extend_bbox = np.concatenate(
        [np.max([[0, 0, 0], bbox[:, 0] - margin], axis=0)[:, np.newaxis],
         np.min([image.shape, bbox[:, 1] + margin], axis=0)[:, np.newaxis]], axis=1)

    rib_bbox = [extend_bbox[0, 0], extend_bbox[0, 1],
                extend_bbox[1, 0], extend_bbox[1, 1],
                extend_bbox[2, 0], extend_bbox[2, 1]]
    if z_num == 1:
        crop_bbox = [
            [rib_bbox[0], rib_bbox[1], rib_bbox[2], rib_bbox[3], rib_bbox[4], x_bbox[0]],
            [rib_bbox[0], rib_bbox[1], rib_bbox[2], rib_bbox[3], x_bbox[1], rib_bbox[5]]
        ]
    elif z_num == 2:
        z_centroid = (rib_bbox[0] + rib_bbox[1]) // 2
        crop_bbox = [
            [rib_bbox[0], min(z_centroid+margin[0], slices), rib_bbox[2], rib_bbox[3], rib_bbox[4], x_bbox[0]],
            [max(0, z_centroid-margin[0]), rib_bbox[1], rib_bbox[2], rib_bbox[3], rib_bbox[4], x_bbox[0]],
            [rib_bbox[0], min(z_centroid + margin[0], slices), rib_bbox[2], rib_bbox[3], x_bbox[1], rib_bbox[5]],
            [max(0, z_centroid - margin[0]), rib_bbox[1], rib_bbox[2], rib_bbox[3], x_bbox[1], rib_bbox[5]],
        ]

    all_crop_image = []
    all_crop_mask = []
    for bbox in crop_bbox:
        crop_image = image[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
        crop_mask = mask[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
        all_crop_image.append(crop_image)
        all_crop_mask.append(crop_mask)

    return all_crop_image, all_crop_mask


def crop_image_by_multi_info(image, rib, x_bbox, z_num=2, margin=10):
    t_mask = rib > 0
    slices, _, _ = image.shape
    zz, yy, xx = np.where(t_mask)

    bbox = np.array([[np.min(zz), np.max(zz)], [np.min(yy), np.max(yy)],
                    [np.min(xx), np.max(xx)]])

    extend_bbox = np.concatenate(
        [np.max([[0, 0, 0], bbox[:, 0] - margin], axis=0)[:, np.newaxis],
         np.min([image.shape, bbox[:, 1] + margin], axis=0)[:, np.newaxis]], axis=1)

    rib_bbox = [extend_bbox[0, 0], extend_bbox[0, 1],
                extend_bbox[1, 0], extend_bbox[1, 1],
                extend_bbox[2, 0], extend_bbox[2, 1]]
    if z_num == 1:
        crop_bbox = [
            [rib_bbox[0], rib_bbox[1], rib_bbox[2], rib_bbox[3], rib_bbox[4], x_bbox[0]],
            [rib_bbox[0], rib_bbox[1], rib_bbox[2], rib_bbox[3], x_bbox[1], rib_bbox[5]]
        ]
    elif z_num == 2:
        z_centroid = (rib_bbox[0] + rib_bbox[1]) // 2
        crop_bbox = [
            [rib_bbox[0], min(z_centroid+margin[0], slices), rib_bbox[2], rib_bbox[3], rib_bbox[4], x_bbox[0]],
            [max(0, z_centroid-margin[0]), rib_bbox[1], rib_bbox[2], rib_bbox[3], rib_bbox[4], x_bbox[0]],
            [rib_bbox[0], min(z_centroid + margin[0], slices), rib_bbox[2], rib_bbox[3], x_bbox[1], rib_bbox[5]],
            [max(0, z_centroid - margin[0]), rib_bbox[1], rib_bbox[2], rib_bbox[3], x_bbox[1], rib_bbox[5]],
        ]

    all_crop_image = []
    all_crop_bbox = []
    for bbox in crop_bbox:
        crop_image = image[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
        all_crop_image.append(crop_image)
        all_crop_bbox.append([bbox[0], bbox[2], bbox[4]])

    return all_crop_image, all_crop_bbox


def crop_mask_by_bbox(mask, bbox):
    """crop image based on bounding box."""
    image_shape = mask.shape
    refine_bbox = [max(0, bbox[0]),
                   min(image_shape[0]-1, bbox[1]),
                   max(0, bbox[2]),
                   min(image_shape[1]-1, bbox[3]),
                   max(0, bbox[4]),
                   min(image_shape[2]-1, bbox[5])]
    crop_mask = mask[refine_bbox[0]:refine_bbox[1],
                     refine_bbox[2]:refine_bbox[3],
                     refine_bbox[4]:refine_bbox[5]]

    return crop_mask


def crop_image_by_bbox(image, bbox):
    """crop image based on bounding box."""
    image_shape = image.shape
    refine_bbox = [max(0, bbox[0]),
                   min(image_shape[0]-1, bbox[1]),
                   max(0, bbox[2]),
                   min(image_shape[1]-1, bbox[3]),
                   max(0, bbox[4]),
                   min(image_shape[2]-1, bbox[5])]
    crop_image = image[refine_bbox[0]:refine_bbox[1],
                       refine_bbox[2]:refine_bbox[3],
                       refine_bbox[4]:refine_bbox[5]]

    return crop_image, refine_bbox


def crop_image_mask_by_bbox(image, mask, bbox):
    """crop image and mask based on bounding box."""
    image_shape = image.shape
    refine_bbox = [max(0, bbox[0]),
                   min(image_shape[0]-1, bbox[1]),
                   max(0, bbox[2]),
                   min(image_shape[1]-1, bbox[3]),
                   max(0, bbox[4]),
                   min(image_shape[2]-1, bbox[5])]
    crop_image = image[refine_bbox[0]:refine_bbox[1],
                       refine_bbox[2]:refine_bbox[3],
                       refine_bbox[4]:refine_bbox[5]]
    crop_mask = mask[refine_bbox[0]:refine_bbox[1],
                     refine_bbox[2]:refine_bbox[3],
                     refine_bbox[4]:refine_bbox[5]]

    return crop_image, crop_mask


def crop_image_mask_by_rib(image=None, mask=None, rib=None, margin=[20, 20, 20]):
    """crop fracture image and mask based on rib mask."""
    t_mask = rib > 0
    zz, yy, xx = np.where(t_mask)

    bbox = np.array([[np.min(zz), np.max(zz)], [np.min(yy), np.max(yy)],
                    [np.min(xx), np.max(xx)]])

    extend_bbox = np.concatenate(
        [np.max([[0, 0, 0], bbox[:, 0] - margin], axis=0)[:, np.newaxis],
         np.min([image.shape, bbox[:, 1] + margin], axis=0)[:, np.newaxis]], axis=1)

    crop_bbox = [extend_bbox[0, 0],
                 extend_bbox[1, 0],
                 extend_bbox[2, 0]]

    image_out = image[extend_bbox[0, 0]: extend_bbox[0, 1],
                      extend_bbox[1, 0]: extend_bbox[1, 1],
                      extend_bbox[2, 0]: extend_bbox[2, 1]]

    mask_out = mask[extend_bbox[0, 0]: extend_bbox[0, 1],
                    extend_bbox[1, 0]: extend_bbox[1, 1],
                    extend_bbox[2, 0]: extend_bbox[2, 1]] if len(mask) != 1 else None

    return image_out, mask_out, crop_bbox


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