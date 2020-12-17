
import sys
sys.path.append("/fileser/zhangfan/LungProject/lung_segment/")
import os
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_processor.data_io import DataIO
from utils.csv_tools import read_csv
from utils.image_utils import crop_image_mask_by_multi_info
from data_processor.data_resample import DataResampler


class GenCropFracData(Dataset):
    def __init__(self, image_dir, rib_mask_dir, ann_dir, centerline_dir,
                 uids_list_path, out_dir, data_source="private", extend_size=10,
                 out_size=[256, 192, 256]):
        self.image_dir = image_dir
        self.rib_mask_dir = rib_mask_dir
        self.ann_dir = ann_dir
        self.centerline_dir = centerline_dir

        self.extend_size = extend_size
        self.out_size = out_size
        self.data_source = data_source

        uids = read_csv(uids_list_path)[1:]
        self.all_uids = [item[0] for item in uids]

        self.out_image_dir = out_dir + "image/"
        self.out_mask_dir = out_dir + "mask/"
        if not os.path.exists(self.out_image_dir):
            os.mkdir(self.out_image_dir)
        if not os.path.exists(self.out_mask_dir):
            os.mkdir(self.out_mask_dir)

    def __len__(self):
        return len(self.all_uids)

    def __getitem__(self, index):
        uid = self.all_uids[index]
        data_loader = DataIO()
        data_resampler = DataResampler()

        # load image dict
        if self.data_source == "private":
            image_path = self.image_dir + uid + ".nii.gz"
        else:
            image_path = self.image_dir + uid + "-image.nii.gz"
        image_dict = data_loader.load_nii_image(image_path)
        image = image_dict["image"]
        spacing = image_dict["spacing"]

        # load rib mask
        if self.data_source == "private":
            rib_mask_path = self.rib_mask_dir + uid + ".mha"
        else:
            rib_mask_path = self.rib_mask_dir + uid + "-image.nii.gz.mha"
        if not os.path.exists(rib_mask_path):
            print("Don't exist the uid: {}".format(uid))
            return False
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

        # load fracture ann
        if self.data_source == "private":
            ann_path = self.ann_dir + uid + ".npz"
            ann_dict = np.load(ann_path)
            fracture_mask = ann_dict["mask"]
            labeled_mask = np.zeros_like(fracture_mask)
            labeled_mask[fracture_mask != 0] = 1
        else:
            ann_path = self.ann_dir + uid + "-label.nii.gz"
            ann_dict = data_loader.load_nii_image(ann_path)
            fracture_mask = ann_dict["image"]
            labeled_mask = np.zeros_like(fracture_mask)
            labeled_mask[fracture_mask != 0] = 1

        # crop image and mask
        all_cropped_image, all_cropped_mask = crop_image_mask_by_multi_info(
            image, labeled_mask, rib_mask, crop_bbox_x, num_z, margin)

        for i in range(len(all_cropped_image)):
            cropped_image = all_cropped_image[i]
            cropped_mask = all_cropped_mask[i]
            # if np.sum(cropped_mask) < 100:
            #     continue

            # resample image and mask
            image_zoom, zoom_factor = data_resampler.resampleImageToFixedSize(cropped_image, self.out_size)
            mask_zoom, _ = data_resampler.resampleMaskToFixedSize(cropped_mask, num_label=1, out_size=self.out_size)

            zoom_factor = [zoom_factor[2], zoom_factor[1], zoom_factor[0]]
            spacing_zoom = [spacing[i] * zoom_factor[i] for i in range(3)]

            out_image_path = self.out_image_dir + uid + "_" + str(i) + ".nii.gz"
            out_mask_path = self.out_mask_dir + uid + "_" + str(i) + ".nii.gz"
            data_loader.save_medical_info_and_data(image_zoom, image_dict["origin"],
                                                   spacing_zoom, image_dict["direction"],
                                                   out_image_path)
            data_loader.save_medical_info_and_data(mask_zoom, image_dict["origin"],
                                                   spacing_zoom, image_dict["direction"],
                                                   out_mask_path)

        return True


if __name__ == "__main__":
    # image_dir = "/fileser/CT_RIB/data/image/res0/"
    # rib_mask_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/alpha_413_centerline/miccai_rib_centerline/rib/"
    # ann_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/alpha_413_centerline/miccai_rib_centerline/fracture/"
    # uids_list_path = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/miccai_all_filename.csv"
    # out_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/refine_miccai_data/"

    image_dir = "/fileser/CT_RIB/data/image/res0/"
    rib_mask_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/alpha_413_centerline/rib_centerline_mask/rib/"
    ann_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/fracture_ann_data/"
    centerline_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/alpha_413_centerline/rib_centerline_mask/result/"
    uids_list_path = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/val_filename.csv"
    out_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/refine_data/crop_192_256_192_res0.7/"

    dataset = GenCropFracData(
        image_dir, rib_mask_dir, ann_dir, centerline_dir,
        uids_list_path, out_dir, data_source="private",
        extend_size=10, out_size=[192, 256, 192])

    train_loader = DataLoader(dataset=dataset, batch_size=1,
                              num_workers=8, shuffle=False)

    for is_success in tqdm(train_loader):
        pass