import sys
sys.path.append("/fileser/zhangfan/LungProject/lung_segment/")
import numpy as np
from tqdm import tqdm
from collections import Counter
from skimage.morphology import label

from data_processor.data_io import DataIO
from utils.csv_tools import read_txt, read_csv, get_data_in_database
from utils.image_utils import trans_physical_2_index

if __name__ == "__main__":
    # gold /fileser/rib_fracture/db/gold_new/rib_gold_2.2.2_raw_spacing
    # /fileser/zhangfan/DataSet/rib_fracture_detection/csv/test_148.csv
    # /fileser/zhangfan/DataSet/pipeline_rib_mask/test_mask/rib/

    # db_data_dir = "/fileser/rib_fracture/db/gold_new/rib_gold_2.2.2_raw_spacing"
    # uid_list_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/test_148.csv"
    # rib_mask_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/test_mask/rib/"

    db_data_dir = "/fileser/rib_fracture/db/train_new_20201130/raw_spacing"
    uid_list_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/rib_train_uids_1592.txt"
    rib_mask_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/alpha_413_centerline/raw_spacing_rib_centerline/rib/"

    all_uid = read_txt(uid_list_dir)
    # all_uid = read_csv(uid_list_dir)[1:]
    # all_uid = [item[0] for item in all_uid]

    data_loader = DataIO()
    all_fracture_num = 0
    filter_num = 0
    for uid in tqdm(all_uid):
        data_dict = get_data_in_database(uid, db_data_dir)
        raw_spacing = data_dict['raw_spacing']
        raw_origin = data_dict['raw_origin']

        rib_mask_path = rib_mask_dir + uid + ".mha"
        rib_mask_dict = data_loader.load_nii_image(rib_mask_path)
        rib_mask = rib_mask_dict["image"]

        mask_size = rib_mask.shape
        all_frac_info = data_dict["frac_info"]
        mask_size = [mask_size[2], mask_size[1], mask_size[0]]

        rib_mask[rib_mask != 0] = 1

        frac_ann_info = []
        frac_label = 1
        for frac_info in all_frac_info:
            frac_site = frac_info["frac_site"]
            frac_type = frac_info["frac_type"]
            if frac_site != 3:
                continue

            all_fracture_num += 1

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
            fracture_bbox_start = [max(0, int(voxel_bbox_start[i])) for i in range(3)]
            voxel_bbox_end = trans_physical_2_index(raw_spacing, raw_origin, physical_bbox_end)
            fracture_bbox_end = [min(mask_size[i]-1, int(voxel_bbox_end[i])) for i in range(3)]

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

            if max_label == 0 or max_num < 25:
                filter_num += 1

    print("the number of filtered fracture: {}/{}".format(filter_num, all_fracture_num))
