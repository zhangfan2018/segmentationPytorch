
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.csv_tools import read_txt, get_data_in_database

uid_list_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/rib_train_uids_1592.txt"
db_data_dir = "/fileser/rib_fracture/db/train_new_20201130/raw_spacing"

all_uid = read_txt(uid_list_dir)
all_candidate_diameter = []
for uid in tqdm(all_uid):
    data_dict = get_data_in_database(uid, db_data_dir)
    raw_spacing = data_dict['raw_spacing']
    raw_origin = data_dict['raw_origin']
    raw_size = data_dict["raw_size"]

    all_frac_info = data_dict["frac_info"]
    for frac_info in all_frac_info:
        physical_diameter = frac_info["physical_diameter"]
        max_diameter = max(physical_diameter)
        all_candidate_diameter.append(max_diameter)

print("the number of candidates is {}".format(len(all_candidate_diameter)))
print("the maximum diameter of candidates is {}".format(np.max(np.array(all_candidate_diameter))))
print("the mean diameter of candidates is {}".format(np.mean(np.array(all_candidate_diameter))))

plt.figure()
plt.hist(all_candidate_diameter, bins=60)
plt.title("the distribution of fracture maximum diameter.")
plt.show()
