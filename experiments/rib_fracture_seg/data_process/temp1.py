
from utils.csv_tools import write_to_csv

filename_csv = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/miccai_all_filename.csv"
train_filename_csv = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/miccai_train_filename.csv"
val_filename_csv = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/miccai_val_filename.csv"

uid = "RibFrac"
all_filename = []
train_filename = []
val_filename = []
for i in range(1, 501):
    filename = uid + str(i)
    all_filename.append(filename)
    if i <= 420:
        train_filename.append(filename)
    else:
        val_filename.append(filename)
write_to_csv(all_filename, filename_csv)
write_to_csv(train_filename, train_filename_csv)
write_to_csv(val_filename, val_filename_csv)


