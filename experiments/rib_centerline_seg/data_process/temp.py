
from data_processor.data_loader import DataIO

image_dir = "/fileser/DATA/IMAGE/DICOM/1.2.840.113619.2.55.3.269126727.996.1519824993.105.3"
res_dir = "/fileser/test/bad_case_result/image/1.2.840.113619.2.55.3.269126727.996.1519824993.105.3.nii.gz"

data_loader = DataIO()
image_dict = data_loader.load_dicom_series(image_dir)
data_loader.save_medical_info_and_data(image_dict["image"], image_dict["origin"],
                                       image_dict["spacing"], image_dict["direction"],
                                       res_dir)