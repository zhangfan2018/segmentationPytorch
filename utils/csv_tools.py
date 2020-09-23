
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
"""

import os
import csv
import sys
import numpy as np


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
        save_csv.writerow([header])
    for content in contents:
        if type(content) != list:
            content = [content]
        save_csv.writerow(content)
    csvfile.close()


def folder_to_csv(file_dir, data_suffix, csv_path, header="seriesUid"):
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
        np.savetxt(myfile, content, delimiter="\n")


def txt_to_csv(txt_path, csv_path, header="seriesUid"):
    """convert .txt file to .csv file"""
    contens = read_txt(txt_path)
    write_to_csv(contens, csv_path, header)


def csv_to_txt(csv_path, txt_path):
    """convert .csv file to .txt file."""
    contents = read_csv(csv_path)
    write_txt(txt_path, contents, mod="w")
