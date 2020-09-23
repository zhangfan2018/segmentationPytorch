"""Interpolation image and mask by linear function.
method:
resampleImageToFixedSize
resampleMaskToFixedSize
resampleImageToFixedSpacing
resampleMaskToFixedSpacing
resampleImageToFixedScale
resampleMaskToFixedScale
"""

import numpy as np
from scipy.ndimage.interpolation import zoom


class DataResampler(object):
    def __init__(self):
        super(DataResampler, self).__init__()

    def resampleImageToFixedSize(self, image, out_size, order=1):
        """resample image to fixed size"""
        """
        Args:
            image(numpy array): image array.
            out_size(list): out image size.
            order(int, optional): The order of the spline interpolation.
        Return:
            image_zoom(numpy array): image array after resample.
            zoom_factor(numpy array, size:[3]): zoom factor.
        """
        scale = np.array(out_size) / image.shape
        zoom_factor = image.shape / np.array(out_size)
        image_zoom = zoom(image, scale, order=order)

        return image_zoom, zoom_factor

    def resampleMaskToFixedSize(self, mask, num_label, out_size, order=1):
        """resample mask to fixed size"""
        scale = np.array(out_size) / mask.shape
        zoom_factor = mask.shape / np.array(out_size)
        mask_zoom = np.zeros_like(mask)
        for i in range(1, num_label+1):
            t_mask = mask.copy()
            t_mask[t_mask != i] = 0
            t_mask[t_mask == i] = 1
            t_mask = zoom(t_mask, scale, order=order)
            t_mask = (t_mask > 0.5).astype(np.uint8)
            mask_zoom[t_mask != 0] = i

        return mask_zoom, zoom_factor

    def resampleImageToFixedSpacing(self, image, ori_spacing, out_spacing, order=1):
        """resample image to fixed spacing"""
        """
        Args:
            image(numpy array): image array.
            ori_spacing(list): out image spacing.
            out_spacing(list): out image spacing.
            order(int, optional): The order of the spline interpolation.
        Return:
            image_zoom(numpy array): image array after resample.
            zoom_factor(numpy array, size:[3]): zoom factor.
        """
        scale = np.array(ori_spacing) / np.array(out_spacing)
        zoom_factor = np.array(out_spacing) / np.array(ori_spacing)
        image_zoom = zoom(image, scale, order=order)

        return image_zoom, zoom_factor

    def resampleMaskToFixedSpacing(self, mask, ori_spacing, out_spacing, num_label, order=1):
        """resample mask to fixed spacing"""
        scale = np.array(ori_spacing) / np.array(out_spacing)
        zoom_factor = np.array(out_spacing) / np.array(ori_spacing)
        mask_zoom = np.zeros_like(mask)
        for i in range(1, num_label+1):
            t_mask = mask.copy()
            t_mask[t_mask != i] = 0
            t_mask[t_mask == i] = 1
            t_mask = zoom(t_mask, scale, order=order)
            t_mask = (t_mask > 0.5).astype(np.uint8)
            mask_zoom[t_mask != 0] = i

        return mask_zoom, zoom_factor

    def resampleImageToFixedScale(self, image, scale, order=1):
        """resample image to fixed scale"""
        image_zoom = zoom(image, scale, order=order)

        return image_zoom

    def resampleMaskToFixedScale(self, mask, scale, num_label, order=1):
        """resmaple mask to fixed scale"""
        mask_zoom = np.zeros_like(mask)
        for i in range(1, num_label+1):
            t_mask = mask.copy()
            t_mask[t_mask != i] = 0
            t_mask[t_mask == i] = 1
            t_mask = zoom(t_mask, scale, order=order)
            t_mask = (t_mask > 0.5).astype(np.uint8)
            mask_zoom[t_mask != 0] = i

        return mask_zoom