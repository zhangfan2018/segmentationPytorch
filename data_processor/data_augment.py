"""Augmentation of image and mask data.
method:
random_flip
random_crop_to_labels
random_rotate3D
random_zoom
random_shift
elastic_transform_3d
"""

import random
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


def flip_axis(img_numpy, axis):
    img_numpy = np.asarray(img_numpy).swapaxes(axis, 0)
    img_numpy = img_numpy[::-1, ...]
    img_numpy = img_numpy.swapaxes(0, axis)
    return img_numpy


def transform_matrix_offset_center_3d(matrix, x, y, z):
    offset_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    return ndimage.interpolation.affine_transform(matrix, offset_matrix)


class DataAugmentor(object):
    def __init__(self):
        super(DataAugmentor, self).__init__()

    def random_flip(self, img_numpy, label=None):
        """rand flip image and mask in 3 axis"""
        axes = [0, 1, 2]
        rand = np.random.randint(0, 3)
        img_numpy = flip_axis(img_numpy, axes[rand])
        img_numpy = np.squeeze(img_numpy)

        if label is None:
            return img_numpy
        else:
            y = flip_axis(label, axes[rand])
            y = np.squeeze(y)
        return img_numpy, y

    def random_crop_to_labels(self, img_numpy, label):
        """Random center crop near the label area"""
        """
        :param img_numpy: 3D medical image modality
        :param label: 3D medical image label
        :return: crop image and label
        """
        target_indexs = np.where(label > 0)
        [img_d, img_h, img_w] = img_numpy.shape
        [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
        Z_min = int(min_D * random.random())
        Y_min = int(min_H * random.random())
        X_min = int(min_W * random.random())

        Z_max = int(img_d - (img_d - max_D) * random.random())
        Y_max = int(img_h - (img_h - max_H) * random.random())
        X_max = int(img_w - (img_w - max_W) * random.random())

        Z_min = int(np.max([0, Z_min]))
        Y_min = int(np.max([0, Y_min]))
        X_min = int(np.max([0, X_min]))

        Z_max = int(np.min([img_d, Z_max]))
        Y_max = int(np.min([img_h, Y_max]))
        X_max = int(np.min([img_w, X_max]))

        return img_numpy[Z_min: Z_max, Y_min: Y_max, X_min: X_max], label[Z_min: Z_max, Y_min: Y_max, X_min: X_max]

    def random_rotate3D(self, img_numpy, mask_numpy, min_angle, max_angle):
        """Return a random rotated array in the same shape"""
        assert img_numpy.ndim == 3, "provide a 3d numpy array"
        assert min_angle < max_angle, "min should be less than max val"
        assert min_angle > -360 or max_angle < 360
        all_axes = [(1, 0), (1, 2), (0, 2)]
        angle = np.random.randint(low=min_angle, high=max_angle + 1)
        axes_random_id = np.random.randint(low=0, high=len(all_axes))
        axes = all_axes[axes_random_id]
        return ndimage.rotate(img_numpy, angle, axes=axes), \
               ndimage.rotate(mask_numpy, angle, axes=axes)

    def random_zoom(self, img_numpy, mask_numpy, min_percentage=0.8, max_percentage=1.1):
        """
        :param img_numpy:
        :param min_percentage:
        :param max_percentage:
        :return: zoom in/out aigmented img
        """
        z = np.random.sample() * (max_percentage - min_percentage) + min_percentage
        zoom_matrix = np.array([[z, 0, 0, 0],
                                [0, z, 0, 0],
                                [0, 0, z, 0],
                                [0, 0, 0, 1]])
        return ndimage.interpolation.affine_transform(img_numpy, zoom_matrix), \
               ndimage.interpolation.affine_transform(mask_numpy, zoom_matrix)

    def random_shift(self, img_numpy, max_percentage=0.2):
        """
        :param img_numpy: 3D medical image modality
        :param max_percentage:
        :return: shift image or label
        """
        dim1, dim2, dim3 = img_numpy.shape
        m1, m2, m3 = int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2)
        d1 = np.random.randint(-m1, m1)
        d2 = np.random.randint(-m2, m2)
        d3 = np.random.randint(-m3, m3)
        return transform_matrix_offset_center_3d(img_numpy, d1, d2, d3)

    def elastic_transform_3d(self, img_numpy, labels=None, alpha=1, sigma=20, c_val=0.0, method="linear"):
        """
        :param img_numpy: 3D medical image modality
        :param labels: 3D medical image labels
        :param alpha: scaling factor of gaussian filter
        :param sigma: standard deviation of random gaussian filter
        :param c_val: fill value
        :param method: interpolation method. supported methods : ("linear", "nearest")
        :return: deformed image and/or label
        """
        assert img_numpy.ndim == 3, 'Wrong img shape, provide 3D img'
        if labels is not None:
            assert img_numpy.shape == labels.shape, "Shapes of img and label do not much!"
        shape = img_numpy.shape

        # Define 3D coordinate system
        coords = np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])

        # Interpolated img
        im_intrps = RegularGridInterpolator(coords, img_numpy,
                                            method=method,
                                            bounds_error=False,
                                            fill_value=c_val)

        # Get random elastic deformations
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                             mode="constant", cval=0.) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                             mode="constant", cval=0.) * alpha
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                             mode="constant", cval=0.) * alpha

        # Define sample points
        x, y, z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        indices = np.reshape(x + dx, (-1, 1)), \
                  np.reshape(y + dy, (-1, 1)), \
                  np.reshape(z + dz, (-1, 1))

        # Interpolate 3D image image
        img_numpy = im_intrps(indices).reshape(shape)

        # Interpolate labels
        if labels is not None:
            lab_intrp = RegularGridInterpolator(coords, labels,
                                                method="nearest",
                                                bounds_error=False,
                                                fill_value=0)

            labels = lab_intrp(indices).reshape(shape).astype(labels.dtype)
            return img_numpy, labels

        return img_numpy