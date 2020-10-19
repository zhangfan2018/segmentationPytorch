
import os
import requests
import subprocess
import pandas as pd

dir1 = r"D:\DataBase\rib_glod_seg_161\csv"
img_list= [os.path.join(dir1,f) for f in os.listdir(dir1) if f.startswith('image_anno')]
caselist = img_list
target_dir = r'D:\DataBase\rib_glod_seg_161\mask'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

# combine all data
df_all = pd.read_csv(caselist[0])
for f in caselist[1:]:
    df = pd.read_csv(f)
    df_all = pd.concat([df_all, df])
print(df_all.shape)

rename_dict = {'影像结果': 'gt_result', '序列编号': 'uid'}
df_all = df_all.rename(columns=rename_dict)
uids = df_all['uid'].drop_duplicates().values
os.chdir(target_dir)

for uid in uids:
    sid_annotations = df_all[df_all['uid'] == uid]
    indices = sid_annotations.index.tolist()
    for i in range(0, len(indices)):
        record = sid_annotations.loc[indices[i]]
        src_address = record.gt_result
        target_path = target_dir + '/%s_%02d.mha' % (uid, i)

        r = requests.get(src_address)
        with open(target_path, 'wb+') as f:
            f.write(r.content)
        f.close()

        print(uid, "finish")