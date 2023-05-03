import base64
import pickle
import json
import numpy as np
import csv
import sys
import zlib
import time
import mmap
from tqdm import tqdm
import joblib
"""
处理 经过Faster R-CNN抽取的coco特征 直接从仓库下载的 tsv文件
"""

# csv.field_size_limit(sys.maxsize)
csv.field_size_limit(2 ** 31 - 1)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
train_img_list = []

# 加载vqa 2.0的训练集 并且获取训练集中的图片ID 这里获取的目的是为了后期划分训练集和验证集
with open('../data/vqa_train.json', 'r') as f:
    vqa_raw = json.load(f)
for raw in vqa_raw.values():
    train_img_list.append(str(raw['image_id']))


def tsv2feature(split):
    if split == 'trainval':
        # infile = '../data/trainval2014_36/trainval2014_resnet101_faster_rcnn_genome_36.tsv'
        infile = 'E:/mukea-out-data/data/trainval2014_36/trainval2014_resnet101_faster_rcnn_genome_36.tsv'
    elif split == 'test':
        # infile = '../data/test2015_36/test2015_resnet101_faster_rcnn_genome_36.tsv'
        infile = 'E:/mukea-out-data/data/test2015_36/test2015_resnet101_faster_rcnn_genome_36.tsv'
    # Verify we can read a tsv
    in_data_train = {}
    in_data_val = {}
    with open(infile, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        # 这里tqdm由于不知道文件的总行数  所以没有进度条
        for item in tqdm(reader):
            # 这里进行数值类型的修改
            item['image_id'] = str(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            w_h = np.array([item['image_w'], item['image_h']])
            # 对指定类型进行读取
            # 其中item['boxes']为对应检测框的位置信息 x,y,w,h  存的大小是 (num_boxes,4)
            # item['features']为对应检测框 在pool5_flat 层的特征
            for field in ['boxes', 'features']:
                # 获取每张图片的所有检测框位置信息 item['boxes'].shape=(num_boxes,4)
                item[field] = np.frombuffer(base64.b64decode(item[field].encode()), dtype=np.float32).reshape(
                    (item['num_boxes'], -1))
            # LXMERT 模型期望这些空间特征是 0 到 1 范围内的归一化边界框 所以这里进行归一化
            spatial_feature = np.concatenate((item['boxes'][:, :2] / w_h, item['boxes'][:, 2:] / w_h), axis=1)
            if item['image_id'] in train_img_list:
                # 这里自己重构存储方式
                in_data_train[item['image_id']] = {'feats': item['features'], 'sp_feats': spatial_feature}
            else:
                in_data_val[item['image_id']] = {'feats': item['features'], 'sp_feats': spatial_feature}
    print(len(in_data_train), len(in_data_val))
    assert len(in_data_train) == 82783
    assert len(in_data_val) == 40504
    # 下面读取处理后的数据进行保存 方便下次直接读取保存的对象
    if split == 'trainval':
        with open('../data/vqa_img_feature_train.pickle', 'wb') as f:
            joblib.dump(in_data_train, f)
        with open('../data/vqa_img_feature_val.pickle', 'wb') as f:
            joblib.dump(in_data_val, f)
    else:
        with open('../data/vqa_img_feature_%s.pickle' % split, 'wb') as f:
            joblib.dump(in_data_val, f)


if __name__ == "__main__":
    tsv2feature('trainval')
    # tsv2feature('val')
