bbox_radius = 32
stride = 48
out_mask = rib_mask.copy()
for i in range(len(all_centerline_coords)):
    centerline_coords = all_centerline_coords[i]
    old_candidate_coords = centerline_coords[0] if len(centerline_coords) else None
    is_new_candidate = True
    for j, coords in enumerate(centerline_coords):
        if j == len(centerline_coords) - 1:
            is_new_candidate = True
        if is_new_candidate:
            candidate_bbox = [int((coords[0] - bbox_radius) / spacing[0]),
                              int((coords[0] + bbox_radius) / spacing[0]),
                              int((coords[1] - bbox_radius) / spacing[1]),
                              int((coords[1] + bbox_radius) / spacing[1]),
                              int((coords[2] - bbox_radius) / spacing[2]),
                              int((coords[2] + bbox_radius) / spacing[2])]
            out_mask[candidate_bbox[4]:candidate_bbox[5],
            candidate_bbox[2]:candidate_bbox[3],
            candidate_bbox[0]:candidate_bbox[1]] = i + 2

        if abs(coords[0] - old_candidate_coords[0]) > stride or \
                abs(coords[1] - old_candidate_coords[1]) > stride or \
                abs(coords[2] - old_candidate_coords[2]) > stride:
            is_new_candidate = True
            old_candidate_coords = coords
        else:
            is_new_candidate = False
out_mask_path = "/fileser/zhangfan/DataSet/rib_fracture_detection/crop_fracture_data/temp/" + uid + ".nii.gz"
data_loader.save_medical_info_and_data(out_mask, image_dict["origin"],
                                       spacing, image_dict["direction"],
                                       out_mask_path)