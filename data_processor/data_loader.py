"""Dataset loader.
"""

import torch
import numpy as np
from torch.utils.data import Dataset

from utils.csv_tools import read_csv
from data_processor.data_io import DataIO
from utils.image_utils import clip_and_normalize
from utils.mask_utils import convert_one_hot, convert_ribCenterline_one_hot


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
        self.file_names = [item[0] for item in file_names]

        if file_exclude_csv:
            exclude_filenames = read_csv(file_exclude_csv)[1:]
            self.exclude_filenames = [item[0] for item in exclude_filenames]

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