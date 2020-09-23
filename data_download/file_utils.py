import json
import os
from datetime import datetime

import SimpleITK as sitk
import numpy as np
import pandas as pd


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)


def read_dicom(folder_name):
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(folder_name)
    dicom_names = reader.GetGDCMSeriesFileNames(folder_name, series_IDs[0])
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image, series_IDs[0]

def load_multi_sheet_df(file,rename_dict=None):
    if file.endswith('.csv'):
        df = pd.read_csv(file, encoding='utf-8',sheetname=None)
    elif file.endswith('.xlsx') or file.endswith('.xls'):
        df = pd.read_excel(file, encoding='utf-8',sheetname=None)
    else:
        print('Bad file %s with invalid format, please check in manual!' % file)
        return None

    temp_list = []
    for key in df.keys():
        temp_list.append(df[key])
    df = pd.concat(temp_list,ignore_index=True)

    if rename_dict is not None:
        df = df.rename(columns=rename_dict)

    df.drop_duplicates(inplace=True)  # 删除重复行
    df.reset_index(drop=True, inplace=True)  # 更新index
    return df 

def load_df(file, rename_dict=None):
    if file.endswith('.csv'):
        df = pd.read_csv(file, encoding='utf-8')
    elif file.endswith('.xlsx') or file.endswith('.xls'):
        df = pd.read_excel(file, encoding='utf-8')
    else:
        print('Bad file %s with invalid format, please check in manual!' % file)
        return None

    if rename_dict is not None:
        df = df.rename(columns=rename_dict)

    df.drop_duplicates(inplace=True)  # 删除重复行
    df.reset_index(drop=True, inplace=True)  # 更新index
    return df


def save_csv(file, data, name=None):
    if not os.path.exists(file):
        os.mknod(file)

    data = pd.DataFrame(columns=name, data=data)
    data.to_csv(file, index=False, encoding='utf-8-sig')


def load_json(json_file):
    # 将文件中的数据读出来
    f = open(json_file, 'r')
    file_data = json.load(f)
    f.close()
    return file_data


def save_json(json_file, dict_data):
    # 将字典保存在filename文件中，并保存在directory文件夹中
    directory = os.path.dirname(json_file)  # 有可能直接给文件名，没有文件夹
    if (directory != '') and (not os.path.exists(directory)):
        os.makedirs(directory)
    f = open(json_file, 'wt')
    json.dump(dict_data, f, cls=MyEncoder, sort_keys=True, indent=4)
    f.close()


def read_txt(txt_file):
    txt_lines = []

    file = open(txt_file, 'r')
    for line in file.readlines():
        line = line.strip()
        txt_lines.append(line)

    return txt_lines


def write_txt(txt_file, data):
    file = open(txt_file, 'w')

    for item in data:
        s = item + '\n'
        file.write(s)
    file.close()


def convert_excel_2_csv(in_file, out_file):
    df = pd.read_excel(in_file, encoding='utf-8')
    df.to_csv(out_file, index=False, encoding='utf-8-sig')


def find_all_files(root, suffix=None):
    res = list()
    for root, _, files in os.walk(root):
        for f in files:
            if suffix is not None and not f.endswith(suffix):
                continue
            res.append(os.path.join(root, f))
    return res
