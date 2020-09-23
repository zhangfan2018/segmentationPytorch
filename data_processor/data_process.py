"""Implementation of preprocess data.
Class:
DataLoaderX
ArgsConfig
PreProcessDataset
LoadData
ProcessOriginalData
ProcessCropData
ProcessCutPatchData
"""

import os
import argparse
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


# data prefetch
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class ArgsConfig:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Data preprocess of lung segmentation')
        # necessary parameter
        parser.add_argument('--csv_path', type=str, default="", help='file to data list in .csv format.')
        parser.add_argument('--image_dir', type=str, default="", help='directory address to the image.')
        parser.add_argument('--mask_dir', type=str, default="", help='directory address to the mask.')
        parser.add_argument('--out_image_dir', type=str, default="", help='directory address to the output image.')
        parser.add_argument('--out_mask_dir', type=str, default="", help='directory address to the output mask.')

        # optional parameter.
        parser.add_argument('--data_suffix', type=str, default=".nii.gz", choices=('.nii.gz', '.nrrd', 'mha'))
        parser.add_argument('--label', type=list, default=[1, 1], help='the label of original and cropped mask.')
        parser.add_argument('--is_smooth_mask', type=bool, default=False, help='whether smoothing original mask')
        parser.add_argument('--extend_size', type=int, default=15,
                            help='the size of extend boundary when crop image and mask.')
        parser.add_argument('--cut_patch_mode', type=str, default="bbox", choices=('bbox', 'centroid'),
                            help="the mode of cutting patch when cut image and mask into patch.")
        parser.add_argument('--is_label1_independent', type=bool, default=False, help='whether label-1 is independent.')
        parser.add_argument('--is_save_smooth_mask', type=bool, default=False, help='whether save smooth mask.')
        parser.add_argument('--is_save_crop_mask', type=bool, default=False, help='whether save cropped mask.')

        # optional parameter, set output size.
        parser.add_argument('--out_ori_size', type=list, default=[],
                            help='resample original image and mask to fixed size.')
        parser.add_argument('--out_ori_spacing', type=list, default=[],
                            help='resample original image and mask to fixed spacing.')
        parser.add_argument('--out_crop_size', type=list, default=[],
                            help='resample cropped image and mask to fixed size.')
        parser.add_argument('--out_crop_spacing', type=list, default=[],
                            help='resample cropped image and mask to fixed spacing.')
        parser.add_argument('--out_crop_patch_size', type=list, default=[],
                            help='resample cropped patch image and mask to fixed size.')
        parser.add_argument('--out_mask_stride', type=list, default=[1],
                            help='resample mask based on stride.')
        self._parser = parser

    def parse(self):
        opt = self._parser.parse_args()
        return opt


# multi thread data preprocess
class PreProcessDataset(Dataset):
    def __init__(self, args):
        file_names = read_csv(args.csv_path)[1:]
        self.file_names = file_names
        self.args = args         # parameter.
        self.data_dict = {}      # temp vars.
        self.list_element = []   # preprocess pipeline element.

    def __len__(self):
        return len(self.file_names)

    def set_pipeline(self, list_element):
        """add preprocess element into list element."""
        self.list_element = list_element

    def execute_pipeline(self):
        """execute preprocess procedure."""
        for element in self.list_element:
            if not element.process(self.args, self.data_dict):
                break

    def __getitem__(self, idx):
        print("the processed number is {}/{}".format(idx, len(self.file_names)))
        uid = self.file_names[idx]
        uid = uid[0] if type(uid) == list else uid
        filename = uid + self.args.data_suffix
        image_path = self.args.image_dir + filename
        mask_path = self.args.mask_dir + filename

        self.data_dict["uid"] = filename
        self.data_dict["image_path"] = image_path
        self.data_dict["mask_path"] = mask_path

        # execute preprocess pipeline procedure
        self.execute_pipeline()

        return True


class LoadData:
    def __init__(self):
        print("load data procedure.")

    def process(self, args, data_dict):
        if not os.path.exists(data_dict["image_path"]):
            print("Don't exist the image path: {}".format(data_dict["image_path"]))
            return False
        elif not os.path.exists(data_dict["mask_path"]):
            print("Don't exist the mask path: {}".format(data_dict["mask_path"]))
            return False

        # load image and mask.
        data_loader = DataIO()
        image_dict = data_loader.load_nii_image(data_dict["image_path"])
        data_dict["image_dict"] = image_dict

        mask_dict = data_loader.load_nii_image(data_dict["mask_path"])
        data_dict["mask_dict"] = mask_dict

        if image_dict["image"].shape != mask_dict["image"].shape:
            print("the shape of image and mask is not the same! the uid: {}".format(data_dict["uid"]))

        return True


class ProcessOriginalData:
    def __init__(self):
        print("process original image and mask.")

    def process(self, args, data_dict):
        image_dict = data_dict["image_dict"]
        mask_dict = data_dict["mask_dict"]
        uid = data_dict["uid"]
        image_zyx = image_dict["image"]
        spacing_ori_t = image_dict["spacing"]
        direction = image_dict["direction"]
        origin = image_dict["origin"]
        spacing_ori = [spacing_ori_t[2], spacing_ori_t[1], spacing_ori_t[0]]
        mask_zyx = mask_dict["image"]
        data_loader = DataIO()

        if args.is_smooth_mask:
            t_smooth_mask = np.zeros_like(mask_zyx)
            for i in range(1, args.label[0] + 1):
                t_mask = mask_zyx.copy()
                if i == 1 and args.is_label1_independent:
                    t_mask[t_mask != 0] = 1
                else:
                    t_mask[t_mask != i] = 0
                    t_mask[t_mask == i] = 1
                if args.is_label1_independent:
                    if i == 1:
                        t_mask = smooth_mask(t_mask, area_least=1000, is_binary_close=True)
                else:
                    t_mask = smooth_mask(t_mask, area_least=1000, is_binary_close=True)
                t_smooth_mask[t_mask != 0] = i
            mask_zyx = t_smooth_mask.copy()

            if args.is_save_smooth_mask:
                if not os.path.exists(args.out_mask_dir + "res0"):
                    os.mkdir(args.out_mask_dir + "res0")
                saved_name = args.out_mask_dir + "res0/" + uid
                data_loader.save_medical_info_and_data(mask_zyx, origin, spacing_ori_t, direction, saved_name)
        data_dict["smooth_mask"] = mask_zyx

        for out_size in args.out_ori_size:
            out_filename = str(out_size[0]) + "_" + str(out_size[1]) + "_" + str(out_size[2])
            if not os.path.exists(args.out_image_dir + "ori_" + out_filename):
                os.mkdir(args.out_image_dir + "ori_" + out_filename)
            if not os.path.exists(args.out_mask_dir + "ori_" + out_filename):
                os.mkdir(args.out_mask_dir + "ori_" + out_filename)

        for out_spacing in args.out_ori_spacing:
            out_filename = str(out_spacing[0]) + "_" + str(out_spacing[1]) + "_" + str(out_spacing[2])
            if not os.path.exists(args.out_image_dir + "ori_res" + out_filename):
                os.mkdir(args.out_image_dir + "ori_res" + out_filename)
            if not os.path.exists(args.out_mask_dir + "ori_res" + out_filename):
                os.mkdir(args.out_mask_dir + "ori_res" + out_filename)

        for out_size in args.out_ori_size:
            out_filename = str(out_size[0]) + "_" + str(out_size[1]) + "_" + str(out_size[2])
            scale = np.array(out_size) / image_zyx.shape
            spacing = np.array(spacing_ori) / scale
            spacing = [spacing[2], spacing[1], spacing[0]]
            image_zoom = zoom(image_zyx, scale, order=1)
            mask_zoom = np.zeros_like(image_zoom)
            for i in range(1, args.label[0] + 1):
                t_mask = mask_zyx.copy()
                if i == 1 and args.is_label1_independent:
                    t_mask[t_mask != 0] = 1
                else:
                    t_mask[t_mask != i] = 0
                    t_mask[t_mask == i] = 1
                t_mask = zoom(t_mask, scale, order=1)
                t_mask = (t_mask > 0.5).astype(np.uint8)
                mask_zoom[t_mask != 0] = i

            saved_name = args.out_image_dir + "ori_" + out_filename + "/" + uid
            data_loader.save_medical_info_and_data(image_zoom, origin, spacing, direction, saved_name)
            saved_name = args.out_mask_dir + "ori_" + out_filename + "/" + uid
            data_loader.save_medical_info_and_data(mask_zoom, origin, spacing, direction, saved_name)

        for out_spacing in args.out_ori_spacing:
            out_filename = str(out_spacing[0]) + "_" + str(out_spacing[1]) + "_" + str(out_spacing[2])
            scale = spacing_ori / np.array(out_spacing)
            image_zoom = zoom(image_zyx, scale, order=1)
            mask_zoom = np.zeros_like(image_zoom)
            for i in range(1, args.label[0]+1):
                mask_tmp = mask_zyx.copy()
                if i == 1 and args.is_label1_independent:
                    mask_tmp[mask_tmp != 0] = 1
                else:
                    mask_tmp[mask_tmp != i] = 0
                    mask_tmp[mask_tmp == i] = 1

                mask_tmp = zoom(mask_tmp, scale, order=1)
                mask_tmp = (mask_tmp > 0.5).astype(np.uint8)
                mask_zoom[mask_tmp != 0] = i
            saved_name = args.out_image_dir + "ori_res" + out_filename + "/" + uid
            data_loader.save_medical_info_and_data(image_zoom, origin, out_spacing, direction, saved_name)
            saved_name = args.out_mask_dir + "ori_res" + out_filename + "/" + uid
            data_loader.save_medical_info_and_data(mask_zoom, origin, out_spacing, direction, saved_name)

        return True


class ProcessCropData:
    def __init__(self):
        print("process crop image and mask.")

    def process(self, args, data_dict):
        image_dict = data_dict["image_dict"]
        uid = data_dict["uid"]
        image_zyx = image_dict["image"]
        spacing_ori_t = image_dict["spacing"]
        direction = image_dict["direction"]
        origin = image_dict["origin"]
        spacing_ori = [spacing_ori_t[2], spacing_ori_t[1], spacing_ori_t[0]]
        mask_zyx = data_dict["smooth_mask"]
        data_loader = DataIO()

        if not args.out_crop_size and not args.out_crop_spacing and \
                not args.out_crop_patch_size and not args.is_save_crop_mask:
            is_crop_image_mask = False
        elif args.out_crop_size or args.out_crop_spacing or \
                args.out_crop_patch_size or args.is_save_crop_mask:
            is_crop_image_mask = True
        else:
            is_crop_image_mask = True

        if is_crop_image_mask:
            margin = [int(args.extend_size / spacing_ori[0]),
                      int(args.extend_size / spacing_ori[1]),
                      int(args.extend_size / spacing_ori[2])]
            crop_image, crop_mask = crop_image_mask(image_zyx, mask_zyx, margin=margin)
            data_dict["crop_image"] = crop_image
            data_dict["crop_mask"] = crop_mask

            if args.is_save_crop_mask:
                if not os.path.exists(args.out_image_dir + "crop_res0"):
                    os.mkdir(args.out_image_dir + "crop_res0")
                if not os.path.exists(args.out_mask_dir + "crop_res0"):
                    os.mkdir(args.out_mask_dir + "crop_res0")
                saved_name = args.out_image_dir + "crop_res0" + "/" + uid
                data_loader.save_medical_info_and_data(crop_image, origin, spacing_ori_t, direction, saved_name)
                saved_name = args.out_mask_dir + "crop_res0" + "/" + uid
                data_loader.save_medical_info_and_data(crop_mask, origin, spacing_ori_t, direction, saved_name)
        else:
            return True

        for out_size in args.out_crop_size:
            out_filename = str(out_size[0]) + "_" + str(out_size[1]) + "_" + str(out_size[2])
            if not os.path.exists(args.out_image_dir + "crop_" + out_filename):
                os.mkdir(args.out_image_dir + "crop_" + out_filename)
            if not os.path.exists(args.out_mask_dir + "crop_" + out_filename):
                os.mkdir(args.out_mask_dir + "crop_" + out_filename)

        for out_spacing in args.out_crop_spacing:
            out_filename = str(out_spacing[0]) + "_" + str(out_spacing[1]) + "_" + str(out_spacing[2])
            if not os.path.exists(args.out_image_dir + "crop_res" + out_filename):
                os.mkdir(args.out_image_dir + "crop_res" + out_filename)
            if not os.path.exists(args.out_mask_dir + "crop_res" + out_filename):
                os.mkdir(args.out_mask_dir + "crop_res" + out_filename)

        for idx, out_size in enumerate(args.out_crop_size):
            out_filename = str(out_size[0]) + "_" + str(out_size[1]) + "_" + str(out_size[2])
            scale = np.array(out_size) / crop_image.shape
            spacing = np.array(spacing_ori) / scale
            spacing = [spacing[2], spacing[1], spacing[0]]
            image_zoom = zoom(crop_image, scale, order=1)
            image_shape = image_zoom.shape
            image_shape = [i//args.out_mask_stride[idx] for i in image_shape]
            mask_scale = np.array(image_shape) / crop_mask.shape
            mask_zoom = np.zeros(image_shape)
            for i in range(1, args.label[1]+1):
                mask_tmp = crop_mask.copy()
                if i == 1 and args.is_label1_independent:
                    mask_tmp[mask_tmp != 0] = 1
                else:
                    mask_tmp[mask_tmp != i] = 0
                    mask_tmp[mask_tmp == i] = 1
                mask_tmp = zoom(mask_tmp, mask_scale, order=1)
                mask_tmp = (mask_tmp > 0.5).astype(np.uint8)
                mask_zoom[mask_tmp != 0] = i

            saved_name = args.out_image_dir + "crop_" + out_filename + "/" + uid
            data_loader.save_medical_info_and_data(image_zoom, origin, spacing, direction, saved_name)
            saved_name = args.out_mask_dir + "crop_" + out_filename + "/" + uid
            data_loader.save_medical_info_and_data(mask_zoom, origin, spacing, direction, saved_name)

        for idx, out_spacing in enumerate(args.out_crop_spacing):
            out_filename = str(out_spacing[0]) + "_" + str(out_spacing[1]) + "_" + str(out_spacing[2])
            scale = spacing_ori / np.array(out_spacing)
            image_zoom = zoom(crop_image, scale, order=1)
            image_shape = image_zoom.shape
            image_shape = [i//args.out_mask_stride[idx] for i in image_shape]
            mask_scale = np.array(image_shape) / crop_mask.shape
            mask_zoom = np.zeros(image_shape)
            for i in range(1, args.label[1]+1):
                mask_tmp = crop_mask.copy()
                if i == 1 and args.is_label1_independent:
                    mask_tmp[mask_tmp != 0] = 1
                else:
                    mask_tmp[mask_tmp != i] = 0
                    mask_tmp[mask_tmp == i] = 1
                mask_tmp = zoom(mask_tmp, mask_scale, order=1)
                mask_tmp = (mask_tmp > 0.5).astype(np.uint8)
                mask_zoom[mask_tmp != 0] = i

            saved_name = args.out_image_dir + "crop_res" + out_filename + "/" + uid
            data_loader.save_medical_info_and_data(image_zoom, origin, out_spacing, direction, saved_name)
            saved_name = args.out_mask_dir + "crop_res" + out_filename + "/" + uid
            data_loader.save_medical_info_and_data(mask_zoom, origin, out_spacing, direction, saved_name)

        return True


class ProcessCutPatchData:
    def __init__(self):
        print("process cut image and mask into patch.")

    def process(self, args, data_dict):
        image_dict = data_dict["image_dict"]
        uid = data_dict["uid"]
        crop_image = image_dict["crop_image"]
        spacing_ori_t = image_dict["spacing"]
        direction = image_dict["direction"]
        origin = image_dict["origin"]
        spacing_ori = [spacing_ori_t[2], spacing_ori_t[1], spacing_ori_t[0]]
        crop_mask = data_dict["crop_mask"]
        data_loader = DataIO()

        for out_size in args.out_crop_patch_size:
            out_filename = str(out_size[0]) + "_" + str(out_size[1]) + "_" + str(out_size[2])
            if not os.path.exists(args.out_image_dir + "crop_patch_" + out_filename):
                os.mkdir(args.out_image_dir + "crop_patch_" + out_filename)
            if not os.path.exists(args.out_mask_dir + "crop_patch_" + out_filename):
                os.mkdir(args.out_mask_dir + "crop_patch_" + out_filename)

        for out_size in args.out_crop_patch_size:
            out_filename = str(out_size[0]) + "_" + str(out_size[1]) + "_" + str(out_size[2])
            if args.cut_patch_mode == "bbox":
                crop_coords = extract_left_right_bbox(crop_mask.copy())
            elif args.cut_patch_mode == "centroid":
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
                for i in range(1, args.label[1] + 1):
                    mask_tmp = t_mask.copy()
                    if i == 1 and args.is_label1_independent:
                        mask_tmp[mask_tmp != 0] = 1
                    else:
                        mask_tmp[mask_tmp != i] = 0
                        mask_tmp[mask_tmp == i] = 1
                    mask_tmp = zoom(mask_tmp, scale, order=1)
                    mask_tmp = (mask_tmp > 0.5).astype(np.uint8)
                    mask_zoom[mask_tmp != 0] = i
                t_uid = uid.split(".nii.gz")[0] + "_{}".format(t_names[idx]) + ".nii.gz"
                saved_name = args.out_image_dir + "crop_patch_" + out_filename + "/" + t_uid
                data_loader.save_medical_info_and_data(image_zoom, origin, spacing, direction, saved_name)
                saved_name = args.out_mask_dir + "crop_patch_" + out_filename + "/" + t_uid
                data_loader.save_medical_info_and_data(mask_zoom, origin, spacing, direction, saved_name)



