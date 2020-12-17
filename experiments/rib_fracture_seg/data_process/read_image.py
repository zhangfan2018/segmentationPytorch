
import os

from data_processor.data_loader import DataIO


data_loader = DataIO()

image_path = "/fileser/zhangfan/DataSet/rib_fracture_detection/refine_data/mask/" \
             "1.2.392.200036.9116.2.5.1.37.2424352877.1520152540.891123.nii.gz"

os.remove(image_path)