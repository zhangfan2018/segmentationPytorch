import sys
sys.path.append("/fileser/zhangfan/LungProject/lung_segment/")
import numpy as np
from tqdm import tqdm
from collections import Counter
from skimage.morphology import label

from data_processor.data_io import DataIO
from utils.csv_tools import read_txt, get_data_in_database
from utils.image_utils import trans_physical_2_index

if __name__ == "__main__":
    # "/fileser/rib_fracture/db/train_new_20201112/raw_spacing"
    # "/fileser/rib_fracture/db/train_new_20201130/raw_spacing"
    db_data_dir = "/fileser/rib_fracture/db/train_new_20201130/raw_spacing"
    uid_list_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/rib_train_uids_1592.txt"
    rib_mask_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/alpha_413_centerline/raw_spacing_rib_centerline/rib/"
    fracture_ann_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/fracture_ann_data/"
    # fracture_mask_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/fracture_mask_temp/"
    # temp_mask_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/temp/"

    all_uid = read_txt(uid_list_dir)
    data_loader = DataIO()
    for uid in tqdm(all_uid):
        data_dict = get_data_in_database(uid, db_data_dir)
        raw_spacing = data_dict['raw_spacing']
        raw_origin = data_dict['raw_origin']

        rib_mask_path = rib_mask_dir + uid + ".mha"
        rib_mask_dict = data_loader.load_nii_image(rib_mask_path)
        rib_mask = rib_mask_dict["image"]

        mask_size = rib_mask.shape
        all_frac_info = data_dict["frac_info"]
        fracture_mask = np.zeros(mask_size, np.uint8)
        mask_size = [mask_size[2], mask_size[1], mask_size[0]]

        rib_mask[rib_mask != 0] = 1
        # out_rib_mask = rib_mask.copy()

        frac_ann_info = []
        frac_label = 1
        for frac_info in all_frac_info:
            frac_site = frac_info["frac_site"]
            frac_type = frac_info["frac_type"]
            # rib_position = frac_info["rib_position"]
            if frac_site != 3:
                continue

            physical_point = frac_info["physical_point"]
            physical_diameter = frac_info["physical_diameter"]
            x1 = physical_point[0] - physical_diameter[0] / 2
            x2 = physical_point[0] + physical_diameter[0] / 2
            y1 = physical_point[1] - physical_diameter[1] / 2
            y2 = physical_point[1] + physical_diameter[1] / 2
            z1 = physical_point[2] - physical_diameter[2] / 2
            z2 = physical_point[2] + physical_diameter[2] / 2
            physical_bbox_start = [x1, y1, z1]
            physical_bbox_end = [x2, y2, z2]

            voxel_bbox_start = trans_physical_2_index(raw_spacing, raw_origin, physical_bbox_start)
            fracture_bbox_start = [max(0, int(voxel_bbox_start[i] - 3/raw_spacing[i])) for i in range(3)]
            voxel_bbox_end = trans_physical_2_index(raw_spacing, raw_origin, physical_bbox_end)
            fracture_bbox_end = [min(mask_size[i]-1, int(voxel_bbox_end[i] + 3/raw_spacing[i])) for i in range(3)]

            fracture_bbox_mask = rib_mask[fracture_bbox_start[2]:fracture_bbox_end[2],
                                          fracture_bbox_start[1]:fracture_bbox_end[1],
                                          fracture_bbox_start[0]:fracture_bbox_end[0]]
            labeled_mask, _ = label(fracture_bbox_mask, neighbors=4, background=0, return_num=True)
            label_count = Counter(labeled_mask.reshape(-1))
            max_num = 0
            max_label = 0
            for key in label_count.keys():
                if key != 0 and label_count[key] > max_num:
                    max_label = key
                    max_num = label_count[key]

            if max_label == 0:
                continue

            temp_mask = np.zeros_like(labeled_mask)
            temp_mask[labeled_mask == max_label] = frac_label

            fracture_mask[fracture_bbox_start[2]:fracture_bbox_end[2],
                          fracture_bbox_start[1]:fracture_bbox_end[1],
                          fracture_bbox_start[0]:fracture_bbox_end[0]] = temp_mask

            candidate_info = [fracture_bbox_start[0], fracture_bbox_end[0],
                              fracture_bbox_start[1], fracture_bbox_end[1],
                              fracture_bbox_start[2], fracture_bbox_end[2],
                              frac_type, frac_label]

            frac_label += 1
            frac_ann_info.append(candidate_info)

            # out_rib_mask[fracture_bbox_start[2]:fracture_bbox_end[2],
            #              fracture_bbox_start[1]:fracture_bbox_end[1],
            #              fracture_bbox_start[0]:fracture_bbox_end[0]] = temp_mask

        frac_ann_path = fracture_ann_dir + uid + ".npz"
        np.savez(frac_ann_path, mask=fracture_mask, annotation=frac_ann_info, spacing=raw_spacing,
                 origin=raw_origin, size=mask_size)

        # fracture_mask_path = fracture_mask_dir + uid + ".nii.gz"
        # data_loader.save_medical_info_and_data(fracture_mask, raw_origin, raw_spacing,
        #                                        rib_mask_dict["direction"], fracture_mask_path)
        #
        # temp_mask_path = temp_mask_dir + uid + ".nii.gz"
        # data_loader.save_medical_info_and_data(out_rib_mask, raw_origin, raw_spacing,
        #                                        rib_mask_dict["direction"], temp_mask_path)
