
import numpy as np


def calculate_iou(bbox1, bbox2):
    """calculate intersection of union between predict and gt bbox."""
    x1, y1, z1 = np.maximum(bbox1[:3], bbox2[:3])
    x2, y2, z2 = np.minimum(bbox1[3:], bbox2[3:])
    intersection = (x2 - x1) * (y2 - y1) * (z2 - z1)
    shape_x1, shape_y1, shape_z1 = bbox1[3:] - bbox1[:3]
    shape_x2, shape_y2, shape_z2 = bbox2[3:] - bbox2[:3]
    union = shape_x1 * shape_y1 * shape_z1 + shape_x2 * shape_y2 * shape_z2 - intersection
    return intersection / union


def generate_candidates_info(all_centerline_coords, stride=48):
    """generate candidates based on centerline coords"""
    all_candidates_centroid = []
    for i in range(len(all_centerline_coords)):
        centerline_coords = all_centerline_coords[i]
        old_candidate_coords = centerline_coords[0] if len(centerline_coords) else None
        is_new_candidate = True
        candidate_type = 1
        for j, coords in enumerate(centerline_coords):
            if j == len(centerline_coords) - 1:
                is_new_candidate = True
                candidate_type = 3
            if is_new_candidate:
                candidate_centroid = [int(coords[0]), int(coords[1]),
                                      int(coords[2]), candidate_type]
                all_candidates_centroid.append(candidate_centroid)

            if abs(coords[0] - old_candidate_coords[0]) > stride or \
               abs(coords[1] - old_candidate_coords[1]) > stride or \
               abs(coords[2] - old_candidate_coords[2]) > stride:
                is_new_candidate = True
                candidate_type = 2
                old_candidate_coords = coords
            else:
                is_new_candidate = False

    return all_candidates_centroid


def generate_candidates(all_centerline_coords, spacing, bbox_radius=32, stride=48):
    """generate candidates based on centerline coords"""
    all_candidates_bbox = []
    for i in range(len(all_centerline_coords)):
        centerline_coords = all_centerline_coords[i]
        old_candidate_coords = centerline_coords[0] if len(centerline_coords) else None
        is_new_candidate = True
        candidate_type = 1
        for j, coords in enumerate(centerline_coords):
            if j == len(centerline_coords) - 1:
                is_new_candidate = True
                candidate_type = 3
            if is_new_candidate:
                candidate_bbox = [int((coords[0] - bbox_radius) / spacing[0]),
                                  int((coords[0] + bbox_radius) / spacing[0]),
                                  int((coords[1] - bbox_radius) / spacing[1]),
                                  int((coords[1] + bbox_radius) / spacing[1]),
                                  int((coords[2] - bbox_radius) / spacing[2]),
                                  int((coords[2] + bbox_radius) / spacing[2]),
                                  candidate_type]
                all_candidates_bbox.append(candidate_bbox)

            if abs(coords[0] - old_candidate_coords[0]) > stride or \
               abs(coords[1] - old_candidate_coords[1]) > stride or \
               abs(coords[2] - old_candidate_coords[2]) > stride:
                is_new_candidate = True
                candidate_type = 2
                old_candidate_coords = coords
            else:
                is_new_candidate = False

    return all_candidates_bbox


def judge_predict_hit_gt(predict_bbox, gt_bbox):
    """judge whether predict bbox hit ground truth bbox"""
    predict_centroid = [(predict_bbox[0] + predict_bbox[1]) / 2,
                        (predict_bbox[2] + predict_bbox[3]) / 2,
                        (predict_bbox[4] + predict_bbox[5]) / 2]

    is_hit = True
    for i in range(3):
        if not (gt_bbox[2 * i] <= predict_centroid[i] <= gt_bbox[2 * i + 1]):
            is_hit = False
            break

    return is_hit


def judge_point_hit_bbox(point, bbox):
    is_hit = True
    for i in range(3):
        if not (bbox[2 * i] <= point[i] <= bbox[2 * i + 1]):
            is_hit = False
            break

    return is_hit