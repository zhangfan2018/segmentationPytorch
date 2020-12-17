
import numpy as np
from tqdm import tqdm

from data_processor.data_io import DataIO
from utils.csv_tools import read_csv, read_txt
from utils.csv_tools import get_data_in_database
from utils.bbox_utils import judge_point_hit_bbox

if __name__ == "__main__":
    db_data_dir = "/fileser/rib_fracture/db/gold_new/rib_gold_2.2.2_raw_spacing"
    uids_list_path = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/test_148.csv"
    centerline_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/alpha_413_centerline/num148_det_thres0.3_ori_result/"

    # db_data_dir = "/fileser/rib_fracture/db/train_new_20201130/raw_spacing"
    # centerline_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/alpha_413_centerline/rib_centerline_mask/result/"
    # uids_list_path = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/rib_train_uids_1592.txt"

    # all_uids = read_txt(uids_list_path)
    all_uids = read_csv(uids_list_path)[1:]
    all_uids = [item[0] for item in all_uids]

    all_candidates_num = 0
    hit_candidates_num = 0
    for uid in tqdm(all_uids):
        data_loader = DataIO()

        # load centerline coords
        all_centerline_coords = []
        all_centerline_path = centerline_dir + uid + "/centerline/"
        for i in range(24):
            centerline_path = all_centerline_path + str(i) + ".csv"
            centerline_coords = read_csv(centerline_path)
            temp_centerline_coords = []
            for coords in centerline_coords:
                temp_coords = coords[0].split(" ")
                out_coords = [float(temp_coords[i]) for i in range(3)]
                temp_centerline_coords.append(out_coords)
            all_centerline_coords.append(temp_centerline_coords)

        data_dict = get_data_in_database(uid, db_data_dir)
        raw_spacing = data_dict['raw_spacing']
        raw_origin = data_dict['raw_origin']
        all_frac_info = data_dict["frac_info"]

        for frac_info in all_frac_info:
            if frac_info["frac_site"] != 3:
                continue

            all_candidates_num += 1
            physical_point = frac_info["physical_point"]
            physical_diameter = frac_info["physical_diameter"]
            x1 = physical_point[0] - physical_diameter[0] / 2 - raw_origin[0]
            x2 = physical_point[0] + physical_diameter[0] / 2 - raw_origin[0]
            y1 = physical_point[1] - physical_diameter[1] / 2 - raw_origin[1]
            y2 = physical_point[1] + physical_diameter[1] / 2 - raw_origin[1]
            z1 = physical_point[2] - physical_diameter[2] / 2 - raw_origin[2]
            z2 = physical_point[2] + physical_diameter[2] / 2 - raw_origin[2]
            gt_bbox = [x1, x2, y1, y2, z1, z2]

            is_hit = False
            for i in range(len(all_centerline_coords)):
                if is_hit:
                    break
                centerline_coords = all_centerline_coords[i]
                for j, coords in enumerate(centerline_coords):
                    if judge_point_hit_bbox(coords, gt_bbox):
                        hit_candidates_num += 1
                        is_hit = True
                        break

    print("the number of hit is {}/{}".format(hit_candidates_num, all_candidates_num))

