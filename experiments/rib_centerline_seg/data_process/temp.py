import shutil

ori_path = "/fileser/CT_RIB/data/image/res0/1.2.392.200036.9116.2.5.1.37.2424352877.1493988287.653677.nii.gz"
dest_path = "/fileser/zhangfan/DataSet/vertebra_location_dataset/bad_case/" \
            "image/1.2.392.200036.9116.2.5.1.37.2424352877.1493988287.653677.nii.gz"

shutil.copy(ori_path, dest_path)