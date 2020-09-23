
import os
import sys
sys.path.append("/fileser/zhangfan/LungProject/segmentPytorch/")
import glob
import shutil
import subprocess
import traceback
from multiprocessing import Pool, cpu_count

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加路径
sys.path.append(BASE_DIR)

from data_download.file_utils import load_multi_sheet_df
from data_download.global_var import DICOM_DIR


class Download(object):
    def __init__(self, url_file):
        if not os.path.exists(DICOM_DIR):
            os.makedirs(DICOM_DIR)

        self.url_file = url_file
        assert os.path.exists(self.url_file)

    def __call__(self):
        pool = Pool(int(cpu_count()))

        rename_dict = {u"序列号": 'seriesInstanceUID',
                       u"文件内网地址": 'urlLAN'}
        df = load_multi_sheet_df(self.url_file, rename_dict)
        series = df['seriesInstanceUID'].drop_duplicates().values
        print('Num of series is %d, num of pools is %d.' % (len(series), int(cpu_count())))

        for uid in series:
            try:
                pool.apply_async(self._single_download, (uid, df))
            except Exception as err:
                traceback.print_exc()
                print('Outer download dicom throws exception %s, with series uid %s!' % (err, uid))

        pool.close()
        pool.join()

    def _single_download(self, uid, df):
        print('Downloading series uid %s.' % uid)

        items = df[df['seriesInstanceUID'] == uid]
        indices = items.index.tolist()

        save_dir = os.path.join(DICOM_DIR, uid)

        # skip download if dcm exists
        if os.path.exists(save_dir):
            num_dcm = len(glob.glob(save_dir + '/*.dcm'))
            if num_dcm != len(items):
                shutil.rmtree(save_dir)
            else:
                return

        try:
            os.makedirs(save_dir)
            for i in range(0, len(indices)):
                record = df.loc[indices[i]]
                url = record.urlLAN

                dicom_name = url.split('.dcm')[0].split('/')[-1] + '.dcm'
                cmd = 'wget -O %s \"%s\"' % (dicom_name, url)
                os.chdir(save_dir)
                subprocess.call(cmd, shell=True)
                os.chdir(DICOM_DIR)
        except Exception as err:
            traceback.print_exc()
            print('Inner download dicom throws exception %s, with series uid %s!' % (err, uid))


if __name__ == '__main__':
    download = Download(
        url_file='/fileser/zhangfan/DataSet/airway_segment_data/csv/data_url_1.xlsx')
    download()
