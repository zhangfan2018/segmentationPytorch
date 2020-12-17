##################################
#         new eval process       #
##################################
import os
import json
import lmdb
import traceback
import numpy as np
import pandas as pd


default_min_diameter = 0
default_max_diameter = 999999


def trans_physical_2_index(spacing, origin, physical_point):
    """convert physical coords to voxel coords."""
    index_point = []
    for i in range(0, 3):
        index_point.append((physical_point[i] - origin[i]) / spacing[i])
    return index_point


def merge_data_list(result_dir, predictions_filename):
    """merge data list from alpha detection result."""
    case_list = [f for f in os.listdir(result_dir) if '.csv' not in f and 'img' not in f]
    name = 'fracture.csv'
    df = pd.DataFrame()
    for f in case_list:
        filepath = os.path.join(result_dir, f, name)
        if os.path.exists(filepath):
            df_new = pd.read_csv(filepath, index_col=0)
            df_new = df_new.drop_duplicates()
            df = pd.concat([df, df_new])
        else:
            print(f)
    df = df.drop_duplicates()
    df.to_csv(predictions_filename)


def csv2df(csv_path, txn):
    """convert prediction result in .csv format to data framework."""
    df = pd.read_csv(csv_path)
    name_dict = {'coordX': 'vx', 'coordY': 'vy', 'coordZ': 'vz',
                 'boneNo': 'bone_no',
                 'boneType': 'bone_type',
                 'frac_type': 'fracture_type',
                 'det_probability': 'probDet'}
    df = df.rename(columns=name_dict)
    uids = df['uid'].drop_duplicates().values
    env_ct_info = lmdb.open(txn, map_size=int(1e9))
    txn_ct_info = env_ct_info.begin()
    temp_list = []
    for uid in uids:
        sid_annotations = df[df['uid'] == uid]
        indices = sid_annotations.index.tolist()

        value = txn_ct_info.get(uid.encode())
        if value is None:
            continue

        value = str(value, encoding='utf-8')

        info_dict = json.loads(value)
        raw_spacing = info_dict['raw_spacing']
        raw_origin = info_dict['raw_origin']
        nii_file = os.path.join(uid + ".nii.gz")
        for j in range(0, len(indices)):
            annotation = sid_annotations.loc[indices[j]]
            ph_z = annotation.vz * raw_spacing[2] + raw_origin[2]
            ph_y = annotation.vy * raw_spacing[1] + raw_origin[1]
            ph_x = annotation.vx * raw_spacing[0] + raw_origin[0]

            diameter_x_px, diameter_y_px, diameter_z_px = annotation.detector_diameterX * raw_spacing[0], \
                                                          annotation.detector_diameterY * raw_spacing[1], \
                                                          annotation.detector_diameterZ * raw_spacing[0]

            bone_type = annotation.bone_type
            bone_no = annotation.bone_no
            fracture_type = annotation.fracture_type

            prob = annotation.probDet

            temp_list.append([
                uid, nii_file,
                ph_z, ph_y, ph_x,
                annotation.vz, annotation.vy, annotation.vx,
                diameter_z_px, diameter_y_px, diameter_x_px,
                raw_spacing[2], raw_spacing[1], raw_spacing[0],
                prob, fracture_type, bone_type, bone_no
            ])

    return pd.DataFrame(
            temp_list,
            columns=[
                "uid", "path",
                "z_px", "y_px", "x_px",
                "vz", "vy", "vx",
                "diameter_z_px", "diameter_y_px", "diameter_x_px",
                "raw_spacing_z", "raw_spacing_y", "raw_spacing_x",
                "probability", "fracture_type", "bone_type", "bone_no"])


def db2df(file):
    """convert gt result in database format to data framework."""
    env = lmdb.open(file)
    txn = env.begin()
    temp_list = []
    for key, value in txn.cursor():
        key = str(key, encoding="utf-8")
        value = str(value, encoding="utf-8")
        dcm_info = json.loads(value)
        # if key in bad_cases_in_test:
        #     continue

        nii_file = "None"
        nodule_infos = dcm_info["frac_info"]
        ori_spacing = dcm_info["raw_spacing"]
        current_spacing = dcm_info['current_spacing']
        for info in nodule_infos:
            if (int(info['frac_site']) != 3):
                continue
            temp_list.append([
                key, nii_file,
                info["physical_point"][2], info["physical_point"][1], info["physical_point"][0],
                info["index_point"][2], info["index_point"][1], info["index_point"][0],
                info["physical_diameter"][2], info["physical_diameter"][1], info["physical_diameter"][0],
                ori_spacing[2], ori_spacing[1], ori_spacing[0], current_spacing[0],
                float(((str(info["frac_type"])).split(',')[0])),
                float((str(info["frac_type"])).split(',')[0]),
                float((str(info["rib_position"])).split(',')[0])])

    env.close()
    temp_df = pd.DataFrame(
        temp_list,
        columns=[
            "uid", "path",
            "z_px", "y_px", "x_px",
            "vz", "vy", "vx",
            "diameter_z_px", "diameter_y_px", "diameter_x_px",
            "raw_spacing_z", "raw_spacing_y", "raw_spacing_x", "current_spacing",
            "fracture_type", "bone_type", "bone_no"])
    return temp_df


class Candidate(object):
    """
    Represents a candidate
    """

    def __init__(self,
                 series_id=None,
                 center_x=None, center_y=None, center_z=None,
                 diameter_x=None, diameter_y=None, diameter_z=None,
                 probability=None, lung_seg=-1, lung_leaf=-1):
        self.series_id = series_id
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.diameter_x = diameter_x
        self.diameter_y = diameter_y
        self.diameter_z = diameter_z
        self.probability = probability
        self.lung_seg = lung_seg
        self.lung_leaf = lung_leaf
        self.nodule_id = None

        self.correspond_predicts = []
        self.correspond_gts = []


def collect_candidate_annotations(annotations, csv_type, series_id_label='uid',
                                  z_label='z_px', y_label='y_px', x_label='x_px', vz = 'vz', vy = 'vy', vx = 'vx',
                                  dz_label='diameter_z_px', dy_label='diameter_y_px', dx_label='diameter_x_px',
                                  bone_label='bone_no', frac_type='fracture_type',
                                  prob_label=None, prob_thresh=0.0, max_predictions=-1):
    """
    Input csv path's column label should at least include
    [z_px, y_px, x_px, diameter_z_px, diameter_y_px, diameter_x_px, uid]
    probability should also be included for the prediction result.

    Args:
        annotations:
        csv_type: 'ground_truth' or 'prediction'. If prediction, the csv must includes the probability column.
        series_id_label:
        z_label:
        y_label:
        x_label:
        dz_label:
        dy_label:
        dx_label:
        prob_label:
        prob_thresh:
        max_predictions: the maximum number of predictions in one scan. -1 or 0 means using all predictions.
    Returns:
        A dict in which keys are series ids and values are nodule list/dict (annotation/prediction phase).
        Elements in the nodule list are nodule instances. Keys are id and values in the nodule dict are nodule ids and
        nodule instances.
    """
    if isinstance(annotations, str):
        annotations = pd.read_csv(annotations)

    annotations = annotations.loc[
        (annotations[dx_label] > default_min_diameter) | (annotations[dy_label] > default_min_diameter)]
    annotations = annotations.loc[
        (annotations[dx_label] <= default_max_diameter) & (annotations[dy_label] <= default_max_diameter)]

    series_ids = np.unique(annotations[[series_id_label]].values)
    column_names = annotations.columns.values.tolist()
    z_col_idx, y_col_idx, x_col_idx, dz_col_idx, dy_col_idx, dx_col_idx = (column_names.index(z_label),
                                                                           column_names.index(y_label),
                                                                           column_names.index(x_label),
                                                                           column_names.index(dz_label),
                                                                           column_names.index(dy_label),
                                                                           column_names.index(dx_label))
    vx_index, vy_index, vz_index = column_names.index(vx), column_names.index(vy), column_names.index(vz)
    label_col_idx = column_names.index(bone_label)
    frac_type_col_idx = column_names.index(frac_type)

    if csv_type == 'prediction':
        prob_col_idx = column_names.index(prob_label)
        annotations = annotations.loc[annotations[prob_label] >= prob_thresh]
        # annotations = annotations.loc[(annotations[dx_label] >= 15) | (annotations[dy_label] >= 15)]

    all_nodules = {}
    nodule_count = 0

    for series_id in series_ids:
        nodules = []
        current_annotations = annotations[annotations[series_id_label] == series_id].values
        if csv_type == 'prediction':
            # sort by probability
            current_annotations = current_annotations[np.argsort(-current_annotations[:, prob_col_idx])]
            if max_predictions > 0:
                current_annotations = current_annotations[:max_predictions]

        for annotation in current_annotations:
            try:
                # create nodule instance
                nodule = Candidate(series_id=series_id,
                                   center_z=annotation[z_col_idx],
                                   center_y=annotation[y_col_idx],
                                   center_x=annotation[x_col_idx],
                                   diameter_z=annotation[dz_col_idx],
                                   diameter_y=annotation[dy_col_idx],
                                   diameter_x=annotation[dx_col_idx],
                                   lung_seg=annotation[label_col_idx],
                                   lung_leaf=annotation[frac_type_col_idx])

                if csv_type == 'prediction':
                    nodule.probability = annotation[prob_col_idx]
                nodule.vz = annotation[vz_index],
                nodule.vy = annotation[vy_index],
                nodule.vx = annotation[vx_index],
                nodules.append(nodule)
            except Exception as err:
                traceback.print_exc()
                print('Collect nodule annotations throws exception %s, with uid %s!' % (err, series_id))

        nodule_count += len(nodules)

        if csv_type == 'prediction':
            nodules_dict = dict()
            for i, nodule in enumerate(nodules):
                nodule.nodule_id = i
                nodules_dict[i] = nodule
            all_nodules[series_id] = nodules_dict
        else:
            all_nodules[series_id] = nodules

    print('Csv type: %s.' % csv_type)
    print('Total number of cts: %d.' % len(all_nodules))
    print('Total number of nidis: %d.' % nodule_count)
    return all_nodules, series_ids


def evaluate(series_ids, prediction_nodules, gt_nodules, diameter_mins, diameter_maxs):
    for series_id in series_ids:
        try:
            current_predicts = prediction_nodules[series_id]
        except KeyError:
            current_predicts = {}

        try:
            current_gts = gt_nodules[series_id]
        except KeyError:
            current_gts = []

        # - loop over the gt nodules to get the attribute 'correspond_predicts' and 'correspond_gts'
        for current_gt in current_gts:
            x_gt = float(current_gt.center_x)
            y_gt = float(current_gt.center_y)
            z_gt = float(current_gt.center_z)
            dx_gt = float(current_gt.diameter_x)
            dy_gt = float(current_gt.diameter_y)
            dz_gt = float(current_gt.diameter_z)

            for key, current_predict in current_predicts.items():
                x_predict = float(current_predict.center_x)
                y_predict = float(current_predict.center_y)
                z_predict = float(current_predict.center_z)

                if ((x_gt - (dx_gt / 2.0)) <= x_predict <= (x_gt + (dx_gt / 2.0))) \
                        and ((y_gt - (dy_gt / 2.0)) <= y_predict <= (y_gt + (dy_gt / 2.0))) \
                        and ((z_gt - (dz_gt / 2.0)) <= z_predict <= (z_gt + (dz_gt / 2.0))):
                    current_predict.correspond_gts.append(current_gt)
                    current_gt.correspond_predicts.append(current_predict)

    # count TP, FP, FN, PRED_TP
    # split by diameter
    for [min_diam, max_diam] in zip(diameter_mins, diameter_maxs):
        tps = []
        fps = []
        fns = []
        pred_tps = []
        print('min diameter: %.1f, max diameter: %.1f.' % (min_diam, max_diam))

        for series_id in series_ids:
            try:
                current_predicts = prediction_nodules[series_id]
            except KeyError:
                current_predicts = {}

            try:
                current_gts = gt_nodules[series_id]
            except KeyError:
                current_gts = []

            for key, current_predict in current_predicts.items():
                if min_diam < max(current_predict.diameter_x, current_predict.diameter_y) <= max_diam:
                    #TODO How to deal with one predict corresponds to more gts?
                    if len(current_predict.correspond_gts) > 0:
                        pred_tps.append([current_predict.center_z,
                                         current_predict.center_y,
                                         current_predict.center_x,
                                         current_predict.diameter_z,
                                         current_predict.diameter_y,
                                         current_predict.diameter_x,
                                         current_predict.vz,
                                         current_predict.vy,
                                         current_predict.vx,
                                         current_predict.probability,
                                         current_predict.lung_seg,
                                         current_gt.lung_leaf,
                                         current_predict.series_id])
                    else:
                        fps.append([current_predict.center_z,
                                    current_predict.center_y,
                                    current_predict.center_x,
                                    current_predict.diameter_z,
                                    current_predict.diameter_y,
                                    current_predict.diameter_x,
                                    current_predict.vz,
                                    current_predict.vy,
                                    current_predict.vx,
                                    current_predict.probability,
                                    current_predict.lung_seg,
                                    current_gt.lung_leaf,
                                    current_predict.series_id])

            for current_gt in current_gts:
                if min_diam < max(current_gt.diameter_x, current_gt.diameter_y) <= max_diam:
                    # TODO How to deal with one gt corresponds to more predicts?
                    if len(current_gt.correspond_predicts) > 0:
                        tps.append([current_gt.center_z,
                                    current_gt.center_y,
                                    current_gt.center_x,
                                    current_gt.diameter_z,
                                    current_gt.diameter_y,
                                    current_gt.diameter_x,
                                    current_gt.vz,
                                    current_gt.vy,
                                    current_gt.vx,
                                    current_gt.lung_seg,
                                    current_gt.correspond_predicts[0].lung_seg,
                                    current_gt.lung_leaf,
                                    current_gt.correspond_predicts[0].lung_leaf,
                                    current_gt.series_id])
                    else:
                        fns.append([current_gt.center_z,
                                    current_gt.center_y,
                                    current_gt.center_x,
                                    current_gt.vz,
                                    current_gt.vy,
                                    current_gt.vx,
                                    current_gt.diameter_z,
                                    current_gt.diameter_y,
                                    current_gt.diameter_x,
                                    current_gt.lung_seg,
                                    current_gt.lung_leaf,
                                    current_gt.series_id])

        print('tps: %d.' % len(tps))
        print('pred tps: %d.' % len(pred_tps))
        print('fps: %d.' % len(fps))
        print('fns: %d.' % len(fns))
        print('recall: %.4f.' % (len(tps) * 1.0 / (len(tps) + len(fns) + 0.000001)))
        print('precision: %.4f' % (len(tps) * 1.0 / (len(tps) + len(fps) +len(pred_tps)-len(tps) + 0.000001)))
        print('-' * 30)

        dst_csv_columns = [u'coordZ', u'coordY', u'coordX', u'diameter_z', u'diameter_y', u'diameter_x',
                           u'vz', u'vy', u'vx', u'gt_label', u'pred_label',u'gt_type', u'pred_type',u'uid']
        df_tps = pd.DataFrame(tps, columns=dst_csv_columns)
        df_tps['label_diff'] = df_tps['pred_label'].astype('int') == df_tps['gt_label'].astype( 'int')
        df_tps['type_diff'] = df_tps['pred_type'].astype('float') == df_tps['gt_type'].astype( 'float')

        print('label acc: ', len(df_tps[df_tps['label_diff']]) / len(df_tps))
        print('type acc: ', len(df_tps[df_tps['type_diff']]) / len(df_tps))
        df_tps.to_csv("/fileser/CT_RIB/static_cpp/algo_gt_label.csv")
        df_fns = pd.DataFrame(fns, columns=[u'coordZ', u'coordY', u'coordX', u'vz', u'vy', u'vx', u'diameter_z', u'diameter_y', u'diameter_x',
                               u'gt_label', u'gt_type', u'uid'])
        df_fns.to_csv("/fileser/CT_RIB/static_cpp/algo_gt_label_fns.csv")

        df_fps = pd.DataFrame(fps, columns=[u'coordZ', u'coordY', u'coordX', u'diameter_z', u'diameter_y', u'diameter_x',u'vz', u'vy', u'vx',
                               u'gt_prob', u'gt_label', u'gt_type', u'uid'])
        df_fps.to_csv("/fileser/CT_RIB/static_cpp/algo_gt_label_fps.csv")


def detect_eval(prediction_df, ground_truth_df, prob_thresh=0):
    diameter_min = [0]
    diameter_max = [default_max_diameter]

    predictions, series_ids_predict = collect_candidate_annotations(
        prediction_df,
        csv_type='prediction',
        prob_label='probability',
        prob_thresh=prob_thresh)

    ground_truths, series_ids_gt = collect_candidate_annotations(
        ground_truth_df,
        csv_type='ground_truth')

    series_ids = list(set(series_ids_gt) & set(series_ids_predict))
    print('series id intersection: %d.' % len(series_ids))
    print('=' * 30)
    evaluate(series_ids, predictions, ground_truths, diameter_min, diameter_max)


if __name__ == '__main__':
    # merge detection result
    # result_dir = '/fileser/zhangfan/DataSet/pipeline_rib_mask/alpha_413_centerline/num148_det_thres0.3_ori_result/'
    # pred_csv_path = os.path.join(result_dir, 'frac_sum.csv')
    # merge_data_list(result_dir, pred_csv_path)

    # convert dataBase to dataFrame
    pred_csv_path = "/fileser/zhangfan/DataSet/rib_fracture_detection/test_result/" \
                    "fracture_cls_seg_temp_16/infer_result.csv"
    gt_db_path = "/fileser/rib_fracture/db/gold_new/rib_gold_2.2.1_raw_spacing"
    df_pred = csv2df(pred_csv_path, gt_db_path)
    df_gt = db2df(gt_db_path)

    detect_eval(df_pred, df_gt, 0)
