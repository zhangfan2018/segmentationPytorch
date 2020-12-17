
import os
import shutil

# "1.2.840.113704.9.1000.16.1.2017100810165690600020003",
# "1.2.392.200036.9116.2.5.1.37.2424352877.1520836043.738066",
# all_uids = [
#     "1.2.392.200036.9116.2.5.1.37.2424352877.1414993813.541262",
#     "1.2.392.200036.9116.2.5.1.37.2424352877.1420769184.868389",
#     "1.2.392.200036.9116.2.5.1.37.2424352877.1423380702.985958",
#     "1.2.392.200036.9116.2.5.1.37.2424352877.1442903977.390917",
#     "1.2.392.200036.9116.2.5.1.37.2424352877.1459215456.818924",
#     "1.2.392.200036.9116.2.5.1.37.2424352877.1468662715.990124",
# ]

all_uids = ["1.2.392.200036.9116.2.5.1.37.2424352877.1395274289.549607"]

ori_image_dir = "/fileser/CT_RIB/data/image/res0/"
ori_mask_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/alpha_413_centerline/num148_det_thres0.3_ori_result/"
out_image_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/bad_case/image/"
out_mask_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/bad_case/mask/"

for file_name in all_uids:
    ori_file_path = ori_image_dir + file_name + ".nii.gz"
    dest_file_path = out_image_dir + file_name + ".nii.gz"
    if os.path.exists(ori_file_path):
        shutil.copy(ori_file_path, dest_file_path)

    # ori_file_path = ori_mask_dir + file_name + "/rib_label.mha"
    # dest_file_path = out_mask_dir + file_name + ".nii.gz"
    # if os.path.exists(ori_file_path):
    #     shutil.copy(ori_file_path, dest_file_path)