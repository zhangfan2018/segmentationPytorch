import os

min_density_4_nodule = 1
max_density_4_nodule = 18

min_density_4_mass = 1
max_density_4_mass = 37

# key: density_type.
# value: 0 stands for fp, while 1 stands for tp.

density_trans_dict_4_nodule = dict()
for i in range(min_density_4_nodule, max_density_4_nodule + 1):
    if 1 <= i <= 7:
        density_trans_dict_4_nodule[i] = 1
    else:
        density_trans_dict_4_nodule[i] = 0

density_trans_dict_4_mass = dict()
for i in range(min_density_4_mass, max_density_4_mass + 1):
    if (1 <= i <= 7) or (i == 26):
        density_trans_dict_4_mass[i] = 1
    else:
        density_trans_dict_4_mass[i] = 0

DATA_DIR = '/fileser/zhangfan/DataSet/airway_segment_data/image_1'

IMAGE_DIR = os.path.join(DATA_DIR, 'IMAGE')
LABEL_DIR = os.path.join(DATA_DIR, 'LABEL')

DICOM_DIR = os.path.join(IMAGE_DIR, 'DICOM')
RAW_NII_DIR = os.path.join(IMAGE_DIR, 'RAW_NII')
RAW_CT_INFO = os.path.join(IMAGE_DIR, 'RAW_CT_INFO')
SPACING_ONE_NII = os.path.join(IMAGE_DIR, 'SPACING_1.0_NII')
SPACING_TWO_NII = os.path.join(IMAGE_DIR, 'SPACING_2.0_NII')
LUNG_CUTTING_SPACING_HALF_NII = os.path.join(IMAGE_DIR, 'LUNG_CUTTING_SPACING_0.5_NII', 'img_data')
LUNG_CUTTING_SPACING_ONE_NII = os.path.join(IMAGE_DIR, 'LUNG_CUTTING_SPACING_1.0_NII', 'img_data')

LABEL_DETECT_DIR = os.path.join(LABEL_DIR, 'detect')

LUNA_DIR = os.path.join(IMAGE_DIR, 'LUNA')
LUNA_RAW_NII_DIR = os.path.join(LUNA_DIR, 'RAW_NII')
LUNA_RAW_CT_INFO = os.path.join(LUNA_DIR, 'RAW_CT_INFO')
LUNA_LABEL_INFO = os.path.join(LUNA_DIR, 'LABEL_INFO')

CSV_DIR_EXPORT_FROM_JIANYING = '/fileser/Label-ing/jianying'
CSV_DIR_CONVERT_FOR_ALG = '/fileser/Label-ing/alg'
