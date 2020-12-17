
"""process csv file tools
method:
open_for_csv
read_csv
write_csv
merge_csv
write_to_csv
folder_to_csv
read_txt
write_txt
txt_to_csv
csv_to_txt
get_data_in_database
"""

import os
import csv
import sys

import json
import lmdb
import pandas as pd

def open_for_csv(path):
    """Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


def read_csv(csv_path):
    """read .csv file to list."""
    with open_for_csv(csv_path) as file:
        csv_reader = csv.reader(file, delimiter=',')
        rows = [row for row in csv_reader]
    return rows


def write_csv(csv_name, content, mul=True, mod="w"):
    """write list to .csv file."""
    with open(csv_name, mod) as myfile:
        mywriter = csv.writer(myfile)
        if mul:
            mywriter.writerows(content)
        else:
            mywriter.writerow(content)


def save_csv(file_name, data, header_name=None):
    if not os.path.exists(file_name):
        os.mknod(file_name)

    data = pd.DataFrame(columns=header_name, data=data)
    data.drop_duplicates(inplace=True)
    data.to_csv(file_name, index=False, encoding='utf-8-sig')


def merge_csv(src1, src2, dst):
    """merge two .csv files."""
    contents_1 = read_csv(src1)
    contents_2 = read_csv(src2)[1:]
    contents_dst = contents_1 + contents_2
    if os.path.exists(dst):
        os.remove(dst)
    write_csv(dst, contents_dst)


def write_to_csv(contents, save_path, header=None):
    """contents: list of list or tuple, need to write to csv."""
    csvfile = open(save_path, "w")
    save_csv = csv.writer(csvfile, delimiter=",")
    if header:
        header = [header] if type(header) != list else header
        save_csv.writerow(header)
    for content in contents:
        content = [content] if type(content) != list else content
        save_csv.writerow(content)
    csvfile.close()


def folder_to_csv(file_dir, csv_path, data_suffix=".npz", header="seriesUid"):
    """convert filenames to .csv file."""
    file_names = os.listdir(file_dir)
    file_names = [file_name.split(data_suffix)[0] for file_name in file_names]
    write_to_csv(file_names, csv_path, header)


def read_txt(file_dir):
    """read .txt file to list."""
    rows = []
    with open(file_dir, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            rows.append(line)
    return rows


def write_txt(txt_path, content, mod="w"):
    """write list to .txt file."""
    with open(txt_path, mod) as myfile:
        for t_content in content:
            t_content += "\n"
            myfile.writelines(t_content)
        myfile.close()


def txt_to_csv(txt_path, csv_path, header="seriesUid"):
    """convert .txt file to .csv file"""
    contens = read_txt(txt_path)
    write_to_csv(contens, csv_path, header)


def csv_to_txt(csv_path, txt_path):
    """convert .csv file to .txt file."""
    contents = read_csv(csv_path)[1:]
    write_txt(txt_path, contents, mod="w")


def get_data_in_database(uid, db_dir):
    """Get data dict from database in .db format."""
    env = lmdb.open(db_dir)
    txn = env.begin()
    value = txn.get(uid.encode())
    if value is None:
        print('Not found uid:', uid)
        return None, None, None
    info = str(value, encoding='utf-8')
    info = json.loads(info)
    env.close()

    return info