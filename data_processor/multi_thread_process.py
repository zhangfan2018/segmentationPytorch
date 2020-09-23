"""Implementation of preprocess data in multi thread mode.
"""

import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.ndimage.interpolation import zoom
from prefetch_generator import BackgroundGenerator

from utils.csv_tools import read_csv
from data_processor.data_io import DataIO
from utils.mask_utils import smooth_mask
from utils.image_utils import crop_image_mask
from utils.mask_utils import extract_left_right_bbox


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class PreProcessDataset(Dataset):
    def __init__(self, csv_path=None, image_dir=None, mask_dir=None, label=[3, 2],
                 is_smooth_mask=False, extend_size=20, cut_patch_mode="bbox", is_label1_independent=False,
                 is_save_smooth_mask=False, is_save_crop_mask=False,
                 out_ori_size=[[128, 128, 128]], out_ori_spacing=[[1, 1, 1]],
                 out_crop_size=[[256, 192, 256]], out_crop_spacing=[[1, 1, 1]], out_crop_patch_size=[[256, 192, 128]],
                 out_image_dir=None, out_mask_dir=None,):
        """
        Args:
            csv_path(str): file to data list in .csv format.
            image_dir(str): directory address to the image.
            mask_dir(str): directory address to the mask.
            label(list): the label of original and cropped mask.
            is_smooth_mask(bool): whether smoothing original mask.
            extend_size(int): the size of extend boundary when crop image and mask.
            cut_patch_mode(str, choice(bbox, centroid)): the mode of cutting patch when cut image and mask into patch.
            is_label1_independent(bool): whether label-1 is independent.
            is_save_smooth_mask(bool): whether save smooth mask.
            is_save_crop_mask(bool): whether save cropped mask.
            out_ori_size(list:list): resample original image and mask to fixed size.
            out_ori_spacing(list:list): resample original image and mask to fixed spacing.
            out_crop_size(list:list): resample cropped image and mask to fixed size.
            out_crop_spacing(list:list): resample cropped image and mask to fixed spacing.
            out_crop_patch_size(list:list): resample cropped patch image and mask to fixed size.
            out_image_dir(str): directory address to the output image.
            out_mask_dir(str): directory address to the output mask.
        """

        file_names = read_csv(csv_path)[1:]
        self.file_names = file_names

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.label = label
        self.out_image_dir = out_image_dir
        self.out_mask_dir = out_mask_dir
        self.is_smooth_mask = is_smooth_mask
        self.extend_size = extend_size
        self.cut_patch_mode = cut_patch_mode
        self.is_label1_independent = is_label1_independent
        self.is_save_smooth_mask = is_save_smooth_mask
        self.is_save_crop_mask = is_save_crop_mask

        self.out_ori_size = out_ori_size
        self.out_ori_spacing = out_ori_spacing
        self.out_crop_size = out_crop_size
        self.out_crop_spacing = out_crop_spacing
        self.out_crop_patch_size = out_crop_patch_size

        if not out_crop_size and not out_crop_spacing and not out_crop_patch_size and not is_save_crop_mask:
            self.is_crop_image_mask = False
        elif out_crop_size or out_crop_spacing or out_crop_patch_size or is_save_crop_mask:
            self.is_crop_image_mask = True
        if not self.is_smooth_mask: self.is_save_smooth_mask = False

        if not os.path.exists(self.out_mask_dir + "res0") and self.is_save_smooth_mask:
            os.mkdir(self.out_mask_dir + "res0")

        if not os.path.exists(self.out_image_dir + "crop_res0") and self.is_save_crop_mask:
            os.mkdir(self.out_image_dir + "crop_res0")
        if not os.path.exists(self.out_mask_dir + "crop_res0") and self.is_save_crop_mask:
            os.mkdir(self.out_mask_dir + "crop_res0")

        for out_size in self.out_ori_size:
            out_filename = str(out_size[0]) + "_" + str(out_size[1]) + "_" + str(out_size[2])
            if not os.path.exists(self.out_image_dir + "ori_" + out_filename):
                os.mkdir(self.out_image_dir + "ori_" + out_filename)
            if not os.path.exists(self.out_mask_dir + "ori_" + out_filename):
                os.mkdir(self.out_mask_dir + "ori_" + out_filename)

        for out_spacing in self.out_ori_spacing:
            out_filename = str(out_spacing[0]) + "_" + str(out_spacing[1]) + "_" + str(out_spacing[2])
            if not os.path.exists(self.out_image_dir + "ori_res" + out_filename):
                os.mkdir(self.out_image_dir + "ori_res" + out_filename)
            if not os.path.exists(self.out_mask_dir + "ori_res" + out_filename):
                os.mkdir(self.out_mask_dir + "ori_res" + out_filename)

        for out_size in self.out_crop_size:
            out_filename = str(out_size[0]) + "_" + str(out_size[1]) + "_" + str(out_size[2])
            if not os.path.exists(self.out_image_dir + "crop_" + out_filename):
                os.mkdir(self.out_image_dir + "crop_" + out_filename)
            if not os.path.exists(self.out_mask_dir + "crop_" + out_filename):
                os.mkdir(self.out_mask_dir + "crop_" + out_filename)

        for out_spacing in self.out_crop_spacing:
            out_filename = str(out_spacing[0]) + "_" + str(out_spacing[1]) + "_" + str(out_spacing[2])
            if not os.path.exists(self.out_image_dir + "crop_res" + out_filename):
                os.mkdir(self.out_image_dir + "crop_res" + out_filename)
            if not os.path.exists(self.out_mask_dir + "crop_res" + out_filename):
                os.mkdir(self.out_mask_dir + "crop_res" + out_filename)

        for out_size in self.out_crop_patch_size:
            out_filename = str(out_size[0]) + "_" + str(out_size[1]) + "_" + str(out_size[2])
            if not os.path.exists(self.out_image_dir + "crop_patch_" + out_filename):
                os.mkdir(self.out_image_dir + "crop_patch_" + out_filename)
            if not os.path.exists(self.out_mask_dir + "crop_patch_" + out_filename):
                os.mkdir(self.out_mask_dir + "crop_patch_" + out_filename)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        print("the processed number is {}/{}".format(idx, len(self.file_names)))
        data_loader = DataIO()

        uid = self.file_names[idx]
        uid = uid[0] if type(uid) == list else uid
        uid = uid.split(".nii.gz")[0] + ".nii.gz"
        image_path = self.image_dir + uid
        mask_path = self.mask_dir + uid
        if not os.path.exists(image_path):
            print("don't exist the image path: {}".format(image_path))
            return False, uid
        elif not os.path.exists(mask_path):
            print("don't exist the mask path: {}".format(mask_path))
            return False, uid

        # load and process image
        image_dict = data_loader.load_nii_image(image_path)
        image_zyx = image_dict["image"]
        spacing_ori_t = image_dict["spacing"]
        direction = image_dict["direction"]
        origin = image_dict["origin"]
        spacing_ori = [spacing_ori_t[2], spacing_ori_t[1], spacing_ori_t[0]]

        # load and process mask
        mask_dict = data_loader.load_nii_image(mask_path)
        mask_zyx = mask_dict["image"]

        if image_zyx.shape != mask_zyx.shape:
            print("the shape of image and mask is not the same! the uid: {}".format(uid))
            return False, uid

        if self.is_smooth_mask:
            t_smooth_mask = np.zeros_like(mask_zyx)
            for i in range(1, self.label[0] + 1):
                t_mask = mask_zyx.copy()
                if i == 1 and self.is_label1_independent:
                    t_mask[t_mask != 0] = 1
                else:
                    t_mask[t_mask != i] = 0
                    t_mask[t_mask == i] = 1
                if self.is_label1_independent:
                    if i == 1:
                        t_mask = smooth_mask(t_mask, area_least=300, is_binary_close=True)
                else:
                    t_mask = smooth_mask(t_mask, area_least=300, is_binary_close=True)
                t_smooth_mask[t_mask != 0] = i
            mask_zyx = t_smooth_mask.copy()

        if self.is_save_smooth_mask:
            saved_name = self.out_mask_dir + "res0/" + uid
            data_loader.save_medical_info_and_data(mask_zyx, origin, spacing_ori_t, direction, saved_name)

        for out_size in self.out_ori_size:
            out_filename = str(out_size[0]) + "_" + str(out_size[1]) + "_" + str(out_size[2])
            scale = np.array(out_size) / image_zyx.shape
            spacing = np.array(spacing_ori) / scale
            spacing = [spacing[2], spacing[1], spacing[0]]
            image_zoom = zoom(image_zyx, scale, order=1)
            mask_zoom = np.zeros_like(image_zoom)
            for i in range(1, self.label[0] + 1):
                t_mask = mask_zyx.copy()
                if i == 1 and self.is_label1_independent:
                    t_mask[t_mask != 0] = 1
                else:
                    t_mask[t_mask != i] = 0
                    t_mask[t_mask == i] = 1
                t_mask = zoom(t_mask, scale, order=1)
                t_mask = (t_mask > 0.5).astype(np.uint8)
                mask_zoom[t_mask != 0] = i

            saved_name = self.out_image_dir + "ori_" + out_filename + "/" + uid
            data_loader.save_medical_info_and_data(image_zoom, origin, spacing, direction, saved_name)
            saved_name = self.out_mask_dir + "ori_" + out_filename + "/" + uid
            data_loader.save_medical_info_and_data(mask_zoom, origin, spacing, direction, saved_name)

        for out_spacing in self.out_ori_spacing:
            out_filename = str(out_spacing[0]) + "_" + str(out_spacing[1]) + "_" + str(out_spacing[2])
            scale = spacing_ori / np.array(out_spacing)
            image_zoom = zoom(image_zyx, scale, order=1)
            mask_zoom = np.zeros_like(image_zoom)
            for i in range(1, self.label[0]+1):
                mask_tmp = mask_zyx.copy()
                if i == 1 and self.is_label1_independent:
                    mask_tmp[mask_tmp != 0] = 1
                else:
                    mask_tmp[mask_tmp != i] = 0
                    mask_tmp[mask_tmp == i] = 1

                mask_tmp = zoom(mask_tmp, scale, order=1)
                mask_tmp = (mask_tmp > 0.5).astype(np.uint8)
                mask_zoom[mask_tmp != 0] = i
            saved_name = self.out_image_dir + "ori_res" + out_filename + "/" + uid
            data_loader.save_medical_info_and_data(image_zoom, origin, out_spacing, direction, saved_name)
            saved_name = self.out_mask_dir + "ori_res" + out_filename + "/" + uid
            data_loader.save_medical_info_and_data(mask_zoom, origin, out_spacing, direction, saved_name)

        if self.is_crop_image_mask:
            margin = [int(self.extend_size / spacing_ori[0]),
                      int(self.extend_size / spacing_ori[1]),
                      int(self.extend_size / spacing_ori[2])]
            crop_image, crop_mask = crop_image_mask(image_zyx, mask_zyx, margin=margin)

            if self.is_save_crop_mask:
                saved_name = self.out_image_dir + "crop_res0" + "/" + uid
                data_loader.save_medical_info_and_data(crop_image, origin, spacing_ori_t, direction, saved_name)
                saved_name = self.out_mask_dir + "crop_res0" + "/" + uid
                data_loader.save_medical_info_and_data(crop_mask, origin, spacing_ori_t, direction, saved_name)

        for out_size in self.out_crop_size:
            out_filename = str(out_size[0]) + "_" + str(out_size[1]) + "_" + str(out_size[2])
            scale = np.array(out_size) / crop_image.shape
            spacing = np.array(spacing_ori) / scale
            spacing = [spacing[2], spacing[1], spacing[0]]
            image_zoom = zoom(crop_image, scale, order=1)
            mask_zoom = np.zeros_like(image_zoom)
            for i in range(1, self.label[1]+1):
                mask_tmp = crop_mask.copy()
                if i == 1 and self.is_label1_independent:
                    mask_tmp[mask_tmp != 0] = 1
                else:
                    mask_tmp[mask_tmp != i] = 0
                    mask_tmp[mask_tmp == i] = 1
                mask_tmp = zoom(mask_tmp, scale, order=1)
                mask_tmp = (mask_tmp > 0.5).astype(np.uint8)
                mask_zoom[mask_tmp != 0] = i

            saved_name = self.out_image_dir + "crop_" + out_filename + "/" + uid
            data_loader.save_medical_info_and_data(image_zoom, origin, spacing, direction, saved_name)
            saved_name = self.out_mask_dir + "crop_" + out_filename + "/" + uid
            data_loader.save_medical_info_and_data(mask_zoom, origin, spacing, direction, saved_name)

        for out_spacing in self.out_crop_spacing:
            out_filename = str(out_spacing[0]) + "_" + str(out_spacing[1]) + "_" + str(out_spacing[2])
            scale = spacing_ori / np.array(out_spacing)
            image_zoom = zoom(crop_image, scale, order=1)
            mask_zoom = np.zeros_like(image_zoom)
            for i in range(1, self.label[1]+1):
                mask_tmp = crop_mask.copy()
                if i == 1 and self.is_label1_independent:
                    mask_tmp[mask_tmp != 0] = 1
                else:
                    mask_tmp[mask_tmp != i] = 0
                    mask_tmp[mask_tmp == i] = 1
                mask_tmp = zoom(mask_tmp, scale, order=1)
                mask_tmp = (mask_tmp > 0.5).astype(np.uint8)
                mask_zoom[mask_tmp != 0] = i

            saved_name = self.out_image_dir + "crop_res" + out_filename + "/" + uid
            data_loader.save_medical_info_and_data(image_zoom, origin, out_spacing, direction, saved_name)
            saved_name = self.out_mask_dir + "crop_res" + out_filename + "/" + uid
            data_loader.save_medical_info_and_data(mask_zoom, origin, out_spacing, direction, saved_name)

        for out_size in self.out_crop_patch_size:
            out_filename = str(out_size[0]) + "_" + str(out_size[1]) + "_" + str(out_size[2])
            if self.cut_patch_mode == "bbox":
                crop_coords = extract_left_right_bbox(crop_mask.copy())
            elif self.cut_patch_mode == "centroid":
                pass
            _, _, slices = crop_image.shape
            left_image, right_image = crop_image[:, :, :crop_coords[0]-5], crop_image[:, :, crop_coords[1]+5:]
            left_mask, right_mask = crop_mask[:, :, :crop_coords[0]-5], crop_mask[:, :, crop_coords[1]+5:]

            crop_patch_image = [left_image, right_image]
            crop_patch_mask = [left_mask, right_mask]
            t_names = ["left", "right"]
            for idx in range(2):
                t_image = crop_patch_image[idx]
                t_mask = crop_patch_mask[idx]

                scale = np.array(out_size) / t_image.shape
                spacing = np.array(spacing_ori) / scale
                spacing = [spacing[2], spacing[1], spacing[0]]
                image_zoom = zoom(t_image, scale, order=1)
                mask_zoom = np.zeros_like(image_zoom)
                for i in range(1, self.label[1] + 1):
                    mask_tmp = t_mask.copy()
                    if i == 1 and self.is_label1_independent:
                        mask_tmp[mask_tmp != 0] = 1
                    else:
                        mask_tmp[mask_tmp != i] = 0
                        mask_tmp[mask_tmp == i] = 1
                    mask_tmp = zoom(mask_tmp, scale, order=1)
                    mask_tmp = (mask_tmp > 0.5).astype(np.uint8)
                    mask_zoom[mask_tmp != 0] = i
                t_uid = uid.split(".nii.gz")[0] + "_{}".format(t_names[idx]) + ".nii.gz"
                saved_name = self.out_image_dir + "crop_patch_" + out_filename + "/" + t_uid
                data_loader.save_medical_info_and_data(image_zoom, origin, spacing, direction, saved_name)
                saved_name = self.out_mask_dir + "crop_patch_" + out_filename + "/" + t_uid
                data_loader.save_medical_info_and_data(mask_zoom, origin, spacing, direction, saved_name)

        return True, uid


if __name__ == "__main__":
    csv_path = "/fileser/zhangfan/DataSet/airway_segment_data/csv/luna_mask.csv"
    image_dir = "/fileser/DATA/IMAGE/LUNA/RAW_NII/"
    mask_dir = "/fileser/zhangfan/DataSet/airway_segment_data/luna_mask_nii/"
    image_out_dir = "/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/image_refine/"
    mask_out_dir = "/fileser/zhangfan/DataSet/airway_segment_data/train_lung_airway_data/mask_refine/"

    dataset = PreProcessDataset(csv_path=csv_path, image_dir=image_dir, mask_dir=mask_dir, label=[3, 2],
                                is_smooth_mask=False, extend_size=20, cut_patch_mode="bbox", is_label1_independent=False,
                                is_save_smooth_mask=False, is_save_crop_mask=False,
                                out_ori_size=[[128, 128, 128]], out_ori_spacing=[[1, 1, 1]],
                                out_crop_size=[], out_crop_spacing=[], out_crop_patch_size=[],
                                out_image_dir=image_out_dir, out_mask_dir=mask_out_dir,)

    train_loader = DataLoaderX(
        dataset=dataset,
        batch_size=1,
        num_workers=8,
        shuffle=False)

    for index, flag, uid in enumerate(train_loader):
        pass


