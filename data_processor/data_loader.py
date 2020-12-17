"""Dataset loader.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset

from utils.csv_tools import read_csv
from data_processor.data_io import DataIO
from utils.image_utils import clip_and_normalize
from utils.mask_utils import convert_one_hot, convert_ribCenterline_one_hot
from utils.image_utils import crop_image_mask_by_rib, crop_image_by_bbox
from data_processor.data_resample import DataResampler
from utils.bbox_utils import generate_candidates_info
from utils.image_utils import crop_image_by_multi_info


class DataSetLoader(Dataset):
    """data loader."""
    def __init__(self, csv_path, image_dir, mask_dir, num_classes=1, phase="train", normalization=True,
                 file_exclude_csv=None, window_level=[-1200, 1200]):
        """
        Args:
            csv_path(str): data .csv file.
            image_dir(str): image dir.
            mask_dir(str): mask dir.
            num_classes(int): number of labels.
            phase(str): train or val.
            normalization(bool): whether image normalization.
            file_exclude_csv(str): csv to file which will be excluded, default(None).
            window_level(list): the level of window in CT HU value.
        """

        self.phase = phase
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.normalization = normalization
        self.window_level = window_level

        file_names = read_csv(csv_path)[1:]
        file_names = [item[0] + ".nii.gz" for item in file_names]

        self.file_names = []
        for file_name in file_names:
            file_path = self.image_dir + file_name
            if os.path.exists(file_path):
                self.file_names.append(file_name)

        if file_exclude_csv:
            exclude_filenames = read_csv(file_exclude_csv)[1:]
            self.exclude_filenames = [item[0] + ".nii.gz" for item in exclude_filenames]

            # remove bad case.
            for file_name in self.file_names:
                if file_name in self.exclude_filenames:
                    self.file_names.remove(file_name)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        """
        Return:
            image(torch tensor): channel first, dims=[c, z, y, x]
            mask(torch tensor): channel first, dims=[c, z, y, x]
        """
        data_loader = DataIO()

        # load image.
        uid = self.file_names[idx]
        uid = uid[0] if type(uid) == list else uid
        image_path = self.image_dir + uid
        image_dict = data_loader.load_nii_image(image_path)
        image_zyx = image_dict["image"]
        image_zyx = clip_and_normalize(image_zyx, min_window=self.window_level[0], max_window=self.window_level[1]) \
                    if self.normalization else image_zyx
        image_czyx = image_zyx[np.newaxis, ]

        # load mask.
        if self.mask_dir:
            mask_path = self.mask_dir + uid
            mask_dict = data_loader.load_nii_image(mask_path)
            mask_zyx = mask_dict["image"]
            # convert rib and centerline mask to ont hot.
            mask_czyx = convert_ribCenterline_one_hot(mask_zyx, 1, self.num_classes)
            # mask_czyx = convert_one_hot(mask_zyx, 1, self.num_classes)

        if self.phase != "test":
            return torch.from_numpy(image_czyx).float(), torch.from_numpy(mask_czyx).float()
        else:
            if self.mask_dir:
                return {"uid": uid, "image": image_czyx[np.newaxis, ], "image_shape_ori": image_dict["image"].shape,
                        "mask": mask_czyx[np.newaxis, ], "is_exist_mask": True,
                        "origin": image_dict["origin"], "spacing": image_dict["spacing"],
                        "direction": image_dict["direction"]}
            else:
                return {"uid": uid, "image": image_czyx[np.newaxis, ], "image_shape_ori": image_dict["image"].shape,
                        "mask": None, "is_exist_mask": False,
                        "origin": image_dict["origin"], "spacing": image_dict["spacing"],
                        "direction": image_dict["direction"]}


class CropDataSetLoader(Dataset):
    """data loader."""
    def __init__(self, csv_path, image_dir, mask_dir, num_classes=1, phase="train", normalization=True,
                 file_exclude_csv=None, window_level=[-1200, 1200]):
        """
        Args:
            csv_path(str): data .csv file.
            image_dir(str): image dir.
            mask_dir(str): mask dir.
            num_classes(int): number of labels.
            phase(str): train or val.
            normalization(bool): whether image normalization.
            file_exclude_csv(str): csv to file which will be excluded, default(None).
            window_level(list): the level of window in CT HU value.
        """

        self.phase = phase
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.normalization = normalization
        self.window_level = window_level

        file_names = read_csv(csv_path)[1:]
        file_names = [item[0] for item in file_names]

        self.file_names = []
        for uid in file_names:
            for i in range(4):
                file_name = uid + "_" + str(i) + ".nii.gz"
                file_path = self.image_dir + file_name
                if os.path.exists(file_path):
                    self.file_names.append(file_name)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        """
        Return:
            image(torch tensor): channel first, dims=[c, z, y, x]
            mask(torch tensor): channel first, dims=[c, z, y, x]
        """
        data_loader = DataIO()

        # load image.
        uid = self.file_names[idx]
        uid = uid[0] if type(uid) == list else uid
        image_path = self.image_dir + uid
        image_dict = data_loader.load_nii_image(image_path)
        image_zyx = image_dict["image"]
        image_zyx = clip_and_normalize(image_zyx, min_window=self.window_level[0], max_window=self.window_level[1]) \
                    if self.normalization else image_zyx
        image_czyx = image_zyx[np.newaxis, ]

        # load mask.
        if self.mask_dir:
            mask_path = self.mask_dir + uid
            mask_dict = data_loader.load_nii_image(mask_path)
            mask_zyx = mask_dict["image"]
            mask_czyx = convert_one_hot(mask_zyx, 1, self.num_classes)

        if self.phase != "test":
            return torch.from_numpy(image_czyx).float(), torch.from_numpy(mask_czyx).float()
        else:
            if self.mask_dir:
                return {"uid": uid, "image": image_czyx[np.newaxis, ], "image_shape_ori": image_dict["image"].shape,
                        "mask": mask_czyx[np.newaxis, ], "is_exist_mask": True,
                        "origin": image_dict["origin"], "spacing": image_dict["spacing"],
                        "direction": image_dict["direction"]}
            else:
                return {"uid": uid, "image": image_czyx[np.newaxis, ], "image_shape_ori": image_dict["image"].shape,
                        "mask": None, "is_exist_mask": False,
                        "origin": image_dict["origin"], "spacing": image_dict["spacing"],
                        "direction": image_dict["direction"]}


class FractureDataset(Dataset):
    """Load rib fracture dateset."""
    def __init__(self, csv_path, image_dir, mask_dir,
                 window_level=[-1200, 1200]):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.extend_size = 10
        self.out_size = [256, 192, 256]
        self.window_level = window_level

        file_names = read_csv(csv_path)[1:]
        self.file_names = [item[0] for item in file_names]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data_loader = DataIO()
        data_resampler = DataResampler()
        uid = self.file_names[idx]

        # load image and mask.
        image_path = self.image_dir + uid + ".nii.gz"
        mask_path = self.mask_dir + uid + ".mha"

        image_dict = data_loader.load_nii_image(image_path)
        image_zyx = image_dict["image"]
        spacing = image_dict["spacing"]
        mask_dict = data_loader.load_nii_image(mask_path)
        mask_zyx = mask_dict["image"]

        margin = [int(self.extend_size / spacing[2]),
                  int(self.extend_size / spacing[1]),
                  int(self.extend_size / spacing[0])]

        cropped_image, _, crop_bbox = crop_image_mask_by_rib(image_zyx, np.array([1]), mask_zyx, margin)
        image_zoom, zoom_factor = data_resampler.resampleImageToFixedSize(cropped_image, self.out_size)
        image_norm = clip_and_normalize(image_zoom,
                                        min_window=self.window_level[0],
                                        max_window=self.window_level[1])

        image_bczyx = image_norm[np.newaxis, np.newaxis, ]

        return {"uid": uid, "image": torch.from_numpy(image_bczyx).float(),
                "crop_bbox": crop_bbox, "zoom_factor": zoom_factor}


class FractureCropPatchDataset(Dataset):
    """Load rib fracture dateset."""
    def __init__(self, csv_path, image_dir,
                 rib_mask_dir, centerline_dir,
                 out_size=[192, 256, 192],
                 window_level=[-1200, 1200]):

        self.image_dir = image_dir
        self.rib_mask_dir = rib_mask_dir
        self.centerline_dir = centerline_dir

        self.extend_size = 10
        self.out_size = out_size
        self.window_level = window_level

        uids = read_csv(csv_path)[1:]
        self.all_uids = [item[0] for item in uids]

    def __len__(self):
        return len(self.all_uids)

    def __getitem__(self, index):
        uid = self.all_uids[index]
        data_loader = DataIO()
        data_resampler = DataResampler()

        # load image dict
        image_path = self.image_dir + uid + ".nii.gz"
        image_dict = data_loader.load_nii_image(image_path)
        image = image_dict["image"]
        spacing = image_dict["spacing"]

        # load rib mask
        rib_mask_path = self.rib_mask_dir + uid + ".mha"
        rib_mask_dict = data_loader.load_nii_image(rib_mask_path)
        rib_mask = rib_mask_dict["image"]
        margin = [int(self.extend_size / spacing[2]),
                  int(self.extend_size / spacing[1]),
                  int(self.extend_size / spacing[0])]

        # load centerline coords
        left_coords = []
        right_coords = []
        left_start_id, left_end_id = 0, 0
        right_start_id, right_end_id = 0, 0
        all_centerline_path = self.centerline_dir + uid + "/centerline/"
        for i in range(24):
            centerline_path = all_centerline_path + str(i) + ".csv"
            centerline_coords = read_csv(centerline_path)
            if len(centerline_coords):
                start_centerline_coords = centerline_coords[0][0].split(" ")
                out_coords = [int(float(start_centerline_coords[i]) / spacing[i]) for i in range(3)]
                if i <= 11:
                    right_start_id = i + 1 if right_start_id == 0 else 0
                    right_end_id = i + 1
                    right_coords.append(out_coords)
                else:
                    left_start_id = i - 11 if left_start_id == 0 else 0
                    left_end_id = i - 11
                    left_coords.append(out_coords)

        if not len(left_coords) or not len(right_coords):
            return False

        left_coords = np.array(left_coords)
        right_coords = np.array(right_coords)

        start_id = min(left_start_id, right_start_id)
        end_id = max(left_end_id, right_end_id)
        left_bbox_max = np.max(left_coords[:, 0])
        right_bbox_min = np.min(right_coords[:, 0])
        crop_bbox_x = [left_bbox_max, right_bbox_min]
        num_z = 2 if (end_id - start_id + 1) > 7 else 1

        # crop image
        all_cropped_image, all_crop_bbox = crop_image_by_multi_info(
            image, rib_mask, crop_bbox_x, num_z, margin)

        # resample image
        out_candidates = []
        for i in range(len(all_cropped_image)):
            cropped_image = all_cropped_image[i]
            crop_bbox = all_crop_bbox[i]
            image_zoom, zoom_factor = data_resampler.resampleImageToFixedSize(cropped_image, self.out_size)
            image_norm = clip_and_normalize(image_zoom,
                                            min_window=self.window_level[0],
                                            max_window=self.window_level[1])

            image_czyx = image_norm[np.newaxis, ]
            candidate = {"image": image_czyx, "crop_bbox": crop_bbox, "candidate_type": -1,
                         "zoom_factor": zoom_factor}
            out_candidates.append(candidate)

        return {"uid": uid, "candidates": out_candidates}


class FracturePatchDataset(Dataset):
    """Load rib fracture dateset."""
    def __init__(self, csv_path, image_dir, centerline_dir,
                 out_spacing=[1, 1, 1],
                 bbox_radius=32, stride=48,
                 window_level=[-1200, 1200]):
        self.image_dir = image_dir
        self.centerline_dir = centerline_dir
        self.out_spacing = out_spacing
        self.bbox_radius = bbox_radius
        self.stride = stride
        self.window_level = window_level

        file_names = read_csv(csv_path)[1:]
        self.file_names = [item[0] for item in file_names]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data_loader = DataIO()
        data_resampler = DataResampler()
        uid = self.file_names[idx]

        # load image dict
        image_path = self.image_dir + uid + ".nii.gz"
        image_dict = data_loader.load_nii_image(image_path)
        image = image_dict["image"]
        spacing = image_dict["spacing"]

        # load centerline coords
        all_centerline_coords = []
        all_centerline_path = self.centerline_dir + uid + "/centerline/"
        for i in range(24):
            centerline_path = all_centerline_path + str(i) + ".csv"
            centerline_coords = read_csv(centerline_path)
            temp_centerline_coords = []
            for coords in centerline_coords:
                temp_coords = coords[0].split(" ")
                out_coords = [float(temp_coords[i]) for i in range(3)]
                temp_centerline_coords.append(out_coords)
            all_centerline_coords.append(temp_centerline_coords)

        # generate candidates based on centerline coords.
        all_candidates_centroid = generate_candidates_info(all_centerline_coords, stride=self.stride)

        ori_spacing = [spacing[2], spacing[1], spacing[0]]
        image_zoom, zoom_factor = data_resampler.resampleImageToFixedSpacing(image, ori_spacing, self.out_spacing)
        out_image_shape = image_zoom.shape

        # get cropped image and mask.
        out_candidates = []
        for i in range(len(all_candidates_centroid)):
            ind_candidate_info = all_candidates_centroid[i]
            bbox = ind_candidate_info[0:3]
            refine_bbox = [int((bbox[2]-self.bbox_radius) / self.out_spacing[2]),
                           int((bbox[2]+self.bbox_radius) / self.out_spacing[2]),
                           int((bbox[1]-self.bbox_radius) / self.out_spacing[1]),
                           int((bbox[1]+self.bbox_radius) / self.out_spacing[1]),
                           int((bbox[0]-self.bbox_radius) / self.out_spacing[0]),
                           int((bbox[0]+self.bbox_radius) / self.out_spacing[0])]
            candidate_type = ind_candidate_info[-1]
            cropped_image, refine_bbox = crop_image_by_bbox(image_zoom, refine_bbox)

            image_norm = clip_and_normalize(cropped_image,
                                            min_window=self.window_level[0],
                                            max_window=self.window_level[1])

            norm_image_shape = image_norm.shape
            padded_image = np.zeros([64, 64, 64], np.float)
            padded_image[:norm_image_shape[0], :norm_image_shape[1], :norm_image_shape[2]] = image_norm

            image_czyx = padded_image[np.newaxis, ]
            candidate = {"image": image_czyx, "crop_bbox": refine_bbox, "candidate_type": candidate_type,
                         "zoom_factor": zoom_factor}
            out_candidates.append(candidate)

        return {"uid": uid, "candidates": out_candidates,
                "image_shape": out_image_shape, "zoom_factor": zoom_factor}


class LabeledDataSetLoader(Dataset):
    """data loader."""
    def __init__(self, csv_path, image_dir, mask_dir, num_classes=1, phase="train", is_sample=False, normalization=True,
                 window_level=[-1200, 1200]):
        """
        Args:
            csv_path(str): data .csv file.
            image_dir(str): image dir.
            mask_dir(str): mask dir.
            num_classes(int): number of labels.
            phase(str): train or val.
            normalization(bool): whether image normalization.
            file_exclude_csv(str): csv to file which will be excluded, default(None).
            window_level(list): the level of window in CT HU value.
        """

        self.phase = phase
        self.is_sample = is_sample
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.normalization = normalization
        self.window_level = window_level

        file_names = read_csv(csv_path)[1:]
        file_names = [item[0] for item in file_names]

        self.file_names = []
        for file_name in file_names:
            file_path = self.image_dir + file_name + ".nii.gz"
            if os.path.exists(file_path):
                self.file_names.append(file_name)

        self.labels = {}
        for file_name in self.file_names:
            label = file_name.split('_')[1].split('-')[1]
            self.labels[file_name] = int(label)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        """
        Return:
            image(torch tensor): channel first, dims=[c, z, y, x]
            mask(torch tensor): channel first, dims=[c, z, y, x]
        """
        data_loader = DataIO()
        uid = self.file_names[idx]
        label = self.labels[uid]

        if self.is_sample:
            return label
        uid += ".nii.gz"

        # load image.
        image_path = self.image_dir + uid
        image_dict = data_loader.load_nii_image(image_path)
        image_zyx = image_dict["image"]
        image_zyx = clip_and_normalize(image_zyx, min_window=self.window_level[0], max_window=self.window_level[1]) \
                    if self.normalization else image_zyx
        image_czyx = image_zyx[np.newaxis, ]

        # load mask.
        mask_path = self.mask_dir + uid
        mask_dict = data_loader.load_nii_image(mask_path)
        mask_zyx = mask_dict["image"]
        mask_czyx = convert_one_hot(mask_zyx, 1, self.num_classes)

        return torch.from_numpy(image_czyx).float(), torch.from_numpy(mask_czyx).float(), label
