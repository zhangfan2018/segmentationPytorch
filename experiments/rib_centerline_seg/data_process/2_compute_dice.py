

import os
import numpy as np

from data_processor.data_io import DataIO
from utils.csv_tools import read_csv, write_csv
from runner.metric import compute_dice


if __name__ == "__main__":
    predict_dir = "/fileser/zhangfan/DataSet/lung_rib_data/alpha_ribCenterlineSeg_twoStage/" \
                  "test_161/rib_pipeline/rib/"
    gt_dir = "/fileser/zhangfan/DataSet/lung_rib_data/test_data/ann_no_verify/mask/"
    csv_path = "/fileser/zhangfan/DataSet/lung_rib_data/test_data/csv/test_161.csv"
    result_path = "/fileser/zhangfan/DataSet/lung_rib_data/test_data/csv/pipeline_result.csv"

    data_loader = DataIO()
    dices = []
    data_names = read_csv(csv_path)[1:]
    contents = ["SeriesUID", "rib_dice"]
    write_csv(result_path, contents, mul=False, mod="w")
    for idx, data_name in enumerate(data_names):
        print("the processed image is {}/{}".format(idx, len(data_names)))

        data_name = data_name[0]
        predict_mask_path = predict_dir + data_name + ".mha"
        gt_mask_path = gt_dir + data_name + "_00.mha"
        if not os.path.exists(predict_mask_path) or not os.path.exists(gt_mask_path):
            print("Don't exist the uid: {}".format(data_name))
            continue

        predict_dict = data_loader.load_nii_image(predict_mask_path)
        predict_mask = predict_dict["image"]
        predict_mask[predict_mask != 0] = 1
        gt_dict = data_loader.load_nii_image(gt_mask_path)
        gt_mask = gt_dict["image"]
        gt_mask[gt_mask != 0] = 1

        dice = compute_dice(predict_mask, gt_mask)
        print("the rib dice: {}".format(dice))
        dices.append(dice)
        contents = [data_name, dice]
        write_csv(result_path, contents, mul=False, mod="a+")

    print("the average dice of the rib is {}".format(np.mean(np.array(dices))))