
"""process mask tools
method:
convert_one_hot
extract_bbox
dilation_mask
erosion_mask
remove_small_connected_object
extract_largest_connected_object
keep_KthLargest_connected_object
smooth_mask
extract_left_right_bbox
"""

import numpy as np
from skimage import measure
from skimage.morphology import label
from scipy.ndimage.morphology import generate_binary_structure, binary_closing, \
                                     binary_erosion, binary_dilation


def convert_one_hot(mask, s_idx, num_classes):
    """Convert mask label into one hot coding."""
    masks = []
    for i_label in range(s_idx, num_classes + s_idx):
        mask_i = mask == i_label
        masks.append(mask_i)
    mask_czyx = np.stack(masks, axis=0)
    mask_czyx = mask_czyx.astype(np.float32)
    return mask_czyx


def convert_ribCenterline_one_hot(mask, s_idx, num_classes):
    """Convert rib and centerline mask into one hot coding."""
    masks = []
    for i_label in range(s_idx, num_classes + s_idx):
        mask_i = mask.copy()
        if i_label == 1:
            mask_i[mask_i != 0] = 1
        else:
            mask_i[mask_i != i_label] = 0
            mask_i[mask_i == i_label] = 1
        masks.append(mask_i)
    mask_czyx = np.stack(masks, axis=0)
    mask_czyx = mask_czyx.astype(np.float32)
    return mask_czyx


def extract_bbox(mask):
    """extract object bbox"""
    t_mask = mask > 0
    zz, yy, xx = np.where(t_mask)

    bbox = np.array([[np.min(zz), np.max(zz)], [np.min(yy), np.max(yy)],
                    [np.min(xx), np.max(xx)]])

    return bbox


def dilation_mask(mask, itrs=2):
    struct = generate_binary_structure(3, 2)
    dilated_mask = binary_dilation(mask, structure=struct, iterations=itrs)
    return dilated_mask


def erosion_mask(mask, itrs=2):
    struct = generate_binary_structure(3, 2)
    erosion_mask = binary_erosion(mask, structure=struct, iterations=itrs)
    return erosion_mask


def remove_small_connected_object(mask, area_least=10):
    """remove small connected object"""
    """
    Args:
        mask(numpy array): mask array.
        area_least(int): remain the connected objects that area exceed this threshold.
    Return:
        res_mask(numpy array): re-define mask array.
    """
    mask[mask != 0] = 1
    labeled_mask, num = label(mask, neighbors=4, background=0, return_num=True)
    region_props = measure.regionprops(labeled_mask)

    res_mask = np.zeros_like(mask)
    for i in range(1, num + 1):
        t_area = region_props[i - 1].area
        if t_area > area_least:
            res_mask[labeled_mask == i] = 1

    return res_mask


def extract_largest_connected_object(mask, area_least=10):
    """extract largest connected object"""
    """
    Args:
        mask(numpy array): mask array.
        area_least(int): remain the connected objects that area exceed this threshold.
    Return:
        res_mask(numpy array): re-define mask array.
        centroid(list, size=3): the centroid of the largest connected object.
    """
    mask[mask != 0] = 1
    labeled_mask, num = label(mask, neighbors=4, background=0, return_num=True)
    region_props = measure.regionprops(labeled_mask)

    max_area = 0
    max_index = 0
    centroid = [0, 0, 0]
    for i in range(1, num+1):
        t_area = region_props[i-1].area
        if t_area > max_area:
            max_area = t_area
            max_index = i
            centroid = region_props[i-1].centroid

    if max_area >= area_least:
        res_mask = labeled_mask == max_index
    else:
        res_mask = np.zeros_like(labeled_mask)
    res_mask = res_mask.astype(np.uint8)

    return res_mask, centroid


def keep_KthLargest_connected_object(mask, kth):
    """keep kth largest connected object"""
    mask[mask != 0] = 1
    labeled_mask, num = label(mask, neighbors=4, background=0, return_num=True)
    region_props = measure.regionprops(labeled_mask)

    areas = {}
    for i in range(1, num + 1):
        t_area = region_props[i - 1].area
        areas[str(i)] = t_area

    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)
    res_mask = np.zeros_like(mask)
    for i in range(kth):
        res_mask[labeled_mask == candidates[i][0]] = 1

    return res_mask


def smooth_mask(mask, area_least=10, is_binary_close=False):
    """smooth mask by remove small connected object and binary closing"""
    """
    Args:
        mask(numpy array): mask array.
        area_least(int): remain the connected objects that area exceed this threshold.
        is_binary_close(bool): whether run binary closing.
    Return:
        mask(numpy array): re-define mask array.
    """
    mask = mask.astype(np.uint8)
    mask = remove_small_connected_object(mask, area_least)
    if is_binary_close:
        struct = generate_binary_structure(3, 2)
        mask = binary_closing(mask, structure=struct, iterations=3)
    mask = mask.astype(np.uint8)

    return mask


def extract_left_right_bbox(mask):
    """extract the left and right lung box"""
    # connected region analysis.
    mask[mask != 0] = 1
    labeled_mask, num = label(mask, neighbors=8, background=0, return_num=True)
    region_props = measure.regionprops(labeled_mask)

    # extract object bbox.
    objects_bbox_min = []
    objects_bbox_max = []
    for i in range(num):
        props = region_props[i]
        bbox = props.bbox
        objects_bbox_min.append(bbox[2])
        objects_bbox_max.append(bbox[5])
    objects_bbox_min.sort()
    objects_bbox_max.sort()

    # find the right boundary of left lung and the left boundary of right lung.
    left_diff = 0
    right_diff = 0
    left_idx = num // 2 + 1
    right_idx = num // 2 - 1
    for i in range(int(num * 0.2), int(num * 0.8)+1):
        diff_min = objects_bbox_min[i] - objects_bbox_min[i - 1]
        diff_max = objects_bbox_max[i] - objects_bbox_max[i - 1]
        if diff_min >= left_diff:
            left_diff = diff_min
            left_idx = i
        if diff_max >= right_diff:
            right_diff = diff_max
            right_idx = i
    res = [objects_bbox_min[left_idx], objects_bbox_max[right_idx-1]]

    return res


def find_rib_bound(objects_centroid, interval_value=10):
    """find the FPs of rib mask along the x axis."""
    """
    Args:
    objects_centroid(dict): eg. {1: 100, ...} key:rib label, value:rib centroid along the x axis.
    interval_value(int): the interval rib of two rib.
    Return:
    out_bound_idx(list): the idx of objects which centroids are out of boundary.
    """
    num = len(objects_centroid)
    sorted_centroid = sorted(objects_centroid.items(), key=lambda item: item[1], reverse=False)
    axis_diff = [sorted_centroid[i][1] - sorted_centroid[i - 1][1] for i in range(1, num)]
    sorted_axis_diff = sorted(np.array(axis_diff))
    axis_diff_median = sorted_axis_diff[int(3 / 4 * num)]
    axis_diff_median = max(axis_diff_median, interval_value)

    low_bound_idx = num
    low_diff_value = 0
    for i in range((num - 1) // 3):
        if axis_diff[i] > axis_diff_median * 3 and axis_diff[i] > low_diff_value:
            low_bound_idx = i
            low_diff_value = axis_diff[i]

    high_bound_idx = 0
    high_diff_value = 0
    for j in range((num - 1) // 3):
        if axis_diff[num - 2 - j] > axis_diff_median * 3 and axis_diff[num - 2 - j] > high_diff_value:
            high_bound_idx = num - 1 - j
            high_diff_value = axis_diff[num - 2 - j]

    out_bound_idx = []
    if low_bound_idx != num:
        out_low_bound_idx = [sorted_centroid[i][0] for i in range(low_bound_idx)]
        out_bound_idx.extend(out_low_bound_idx)
    if high_bound_idx != 0:
        out_high_bound_idx = [sorted_centroid[i][0] for i in range(high_bound_idx, num)]
        out_bound_idx.extend(out_high_bound_idx)

    return out_bound_idx, axis_diff_median
