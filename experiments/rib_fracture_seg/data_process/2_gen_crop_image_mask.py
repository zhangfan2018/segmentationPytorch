
import sys
sys.path.append("/fileser/zhangfan/LungProject/lung_segment/")
import os
import copy
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_processor.data_io import DataIO
from utils.csv_tools import read_csv, read_txt
from data_processor.data_resample import DataResampler
from utils.image_utils import crop_image_mask_by_bbox, crop_mask_by_bbox
from utils.bbox_utils import generate_candidates, judge_predict_hit_gt


class GenCropFracData(Dataset):
    def __init__(self, image_dir, rib_mask_dir, centerline_dir, ann_dir,
                 uids_list_path, out_dir, bbox_radius=32, stride=48, neg_pos_ratio=1,
                 iou_thres=0.5, out_size=[64, 64, 64]):
        self.image_dir = image_dir
        self.rib_mask_dir = rib_mask_dir
        self.centerline_dir = centerline_dir
        self.ann_dir = ann_dir

        self.out_size = out_size
        self.bbox_radius = bbox_radius
        self.stride = stride
        self.neg_pos_ratio = neg_pos_ratio
        self.iou_thres = iou_thres

        self.all_uids = read_txt(uids_list_path)
        # uids = read_csv(uids_list_path)[1:]
        # self.all_uids = [item[0] for item in uids]

        self.out_image_dir = out_dir + "image/"
        self.out_mask_dir = out_dir + "mask/"
        if not os.path.exists(self.out_image_dir):
            os.mkdir(self.out_image_dir)
        if not os.path.exists(self.out_mask_dir):
            os.mkdir(self.out_mask_dir)

    def __len__(self):
        return len(self.all_uids)

    def __getitem__(self, index):
        uid = self.all_uids[index]
        data_loader = DataIO()
        data_resampler = DataResampler()

        # load image dict
        image_path = self.image_dir + uid + ".nii.gz"
        image_dict = data_loader.load_nii_image(image_path)
        image = image_dict["image"]
        spacing = image_dict["spacing"]

        # load rib mask
        # rib_mask_path = self.rib_mask_dir + uid + ".mha"
        # rib_mask_dict = data_loader.load_nii_image(rib_mask_path)
        # rib_mask = rib_mask_dict["image"]

        # load centerline coords
        all_centerline_coords = []
        all_centerline_path = self.centerline_dir + uid + "/centerline/"
        for i in range(24):
            centerline_path = all_centerline_path + str(i) + ".csv"
            centerline_coords = read_csv(centerline_path)
            temp_centerline_coords = []
            for coords in centerline_coords:
                temp_coords = coords[0].split(" ")
                # out_coords = [int(float(temp_coords[i]) / spacing[i]) for i in range(3)]
                out_coords = [float(temp_coords[i]) for i in range(3)]
                temp_centerline_coords.append(out_coords)
            all_centerline_coords.append(temp_centerline_coords)

        # load fracture ann
        ann_path = self.ann_dir + uid + ".npz"
        ann_dict = np.load(ann_path)
        fracture_mask = ann_dict["mask"]
        ann_info = ann_dict["annotation"]
        all_fracture_bbox = []
        all_fracture_label = []
        for frac_info in ann_info:
            all_fracture_bbox.append(frac_info[0:6])
            all_fracture_label.append(frac_info[-1])

        # generate candidates based on centerline coords.
        all_candidates = generate_candidates(all_centerline_coords, spacing,
                                             bbox_radius=self.bbox_radius, stride=self.stride)
        print("\nthe number of all candidates is {}".format(len(all_candidates)))

        positive_candidates = []
        negative_candidates = []
        # divide positive and negative candidates.
        for i in range(len(all_candidates)):
            candidate_bbox = all_candidates[i][0:6]
            candidate_type = all_candidates[i][-1]

            is_positive = 0
            hit_label = -1
            hit_bbox = None
            for j in range(len(all_fracture_bbox)):
                gt_bbox = all_fracture_bbox[j]
                if judge_predict_hit_gt(candidate_bbox, gt_bbox):
                    hit_label = all_fracture_label[j]
                    hit_bbox = gt_bbox
                    break

            if hit_label != -1:
                bbox = [candidate_bbox[4], candidate_bbox[5],
                        candidate_bbox[2], candidate_bbox[3],
                        candidate_bbox[0], candidate_bbox[1]]
                candidate_mask = crop_mask_by_bbox(fracture_mask, bbox)
                temp = candidate_mask == hit_label
                temp = temp.astype(np.uint8)
                candidate_voxel_num = np.sum(temp)

                bbox = [hit_bbox[4], hit_bbox[5],
                        hit_bbox[2], hit_bbox[3],
                        hit_bbox[0], hit_bbox[1]]
                gt_mask = crop_mask_by_bbox(fracture_mask, bbox)
                temp = gt_mask == hit_label
                temp = temp.astype(np.uint8)
                gt_voxel_num = np.sum(temp)

                if candidate_voxel_num > gt_voxel_num * self.iou_thres:
                    is_positive = 1

            candidate_info = [candidate_bbox[4], candidate_bbox[5],
                              candidate_bbox[2], candidate_bbox[3],
                              candidate_bbox[0], candidate_bbox[1],
                              hit_label]
            if is_positive == 1:
                positive_candidates.append(candidate_info)
            else:
                negative_candidates.append(candidate_info)

        # sample negative candidates
        # positive:negative = 1:1
        out_candidates = copy.deepcopy(positive_candidates)

        if len(negative_candidates) != 0 and len(positive_candidates) != 0:
            if len(negative_candidates) > len(positive_candidates) * self.neg_pos_ratio:
                sample_negative_idx = np.random.choice(len(negative_candidates),
                                                       len(positive_candidates) * self.neg_pos_ratio,
                                                       replace=False)
            else:
                sample_negative_idx = np.random.choice(len(negative_candidates),
                                                       len(positive_candidates) * self.neg_pos_ratio,
                                                       replace=True)

            for idx in sample_negative_idx:
                out_candidates.append(negative_candidates[idx])
            print("the number of sample candidates is {}".format(len(out_candidates)))

        candidate_cnt = 1
        # get cropped image and mask.
        for i in range(len(out_candidates)):
            ind_candidate_info = out_candidates[i]
            bbox = ind_candidate_info[0:6]
            hit_label = ind_candidate_info[-1]
            cropped_image, cropped_mask = crop_image_mask_by_bbox(image, fracture_mask, bbox)
            out_mask = np.zeros_like(cropped_mask)
            if hit_label != -1:
                out_mask[cropped_mask == hit_label] = 1
            is_positive = 1 if hit_label != -1 else 0

            # resample image and mask
            image_zoom, zoom_factor = data_resampler.resampleImageToFixedSize(cropped_image, self.out_size)
            mask_zoom, _ = data_resampler.resampleMaskToFixedSize(out_mask, num_label=1, out_size=self.out_size)

            zoom_factor = [zoom_factor[2], zoom_factor[1], zoom_factor[0]]
            spacing_zoom = [spacing[i] * zoom_factor[i] for i in range(3)]

            out_image_path = self.out_image_dir + uid + "_" + str(candidate_cnt) + \
                             "-" + str(is_positive) + "_" + ".nii.gz"
            out_mask_path = self.out_mask_dir + uid + "_" + str(candidate_cnt) + \
                            "-" + str(is_positive) + "_" + ".nii.gz"
            candidate_cnt += 1
            data_loader.save_medical_info_and_data(image_zoom, image_dict["origin"],
                                                   spacing_zoom, image_dict["direction"],
                                                   out_image_path)
            data_loader.save_medical_info_and_data(mask_zoom, image_dict["origin"],
                                                   spacing_zoom, image_dict["direction"],
                                                   out_mask_path)
        return True


if __name__ == "__main__":
    image_dir = "/fileser/CT_RIB/data/image/res0/"
    rib_mask_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/alpha_413_centerline/rib_centerline_mask/rib/"
    centerline_dir = "/fileser/zhangfan/DataSet/pipeline_rib_mask/alpha_413_centerline/rib_centerline_mask/result/"
    ann_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/fracture_ann_data/"
    uids_list_path = "/fileser/zhangfan/DataSet/rib_fracture_detection/csv/rib_train_uids_1592.txt"
    out_dir = "/fileser/zhangfan/DataSet/rib_fracture_detection/crop_fracture_data/crop64_res0.75_patch/"

    dataset = GenCropFracData(image_dir, rib_mask_dir, centerline_dir, ann_dir,
                              uids_list_path, out_dir, bbox_radius=24, stride=32,
                              neg_pos_ratio=6, iou_thres=0.3, out_size=[64, 64, 64])

    train_loader = DataLoader(dataset=dataset, batch_size=1,
                              num_workers=8, shuffle=False)

    for is_success in tqdm(train_loader):
        pass










