#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: ZK
# Time: 2019/09/19 19:20
import os
import json
import cv2
import shutil

class_need = ['person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']
std_class_ind = {
    'person': 'person',
    'rider': 'rider',
    'car': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'caravan': 'truck',
    'trailer': 'truck',
    'train': 'train',
    'motorcycle': 'bike',
    'bicycle': 'bike'
}
class_label_id = {'person': 24,
                  'rider': 25,
                  'car': 26,
                  'truck': 27,
                  'bus': 28,
                  'caravan': 29,
                  'trailer': 30,
                  'train': 31,
                  'motorcycle': 32,
                  'bicycle': 33}

ROOT_DIR = os.path.abspath('')  # Root directory of the project
DEFAULT_SAVE_DIR = os.path.join(ROOT_DIR, "cityscapes")


def label_proc(dataset='train'):
    original_img = os.path.join('G:\\Dataset\\Cityscape\\cityscapes\\leftImg8bit', dataset)  # 原始图片的存放位置
    original_label = os.path.join('G:\\Dataset\\Cityscape\\cityscaps_label\\gtFine', dataset)  # 原始标签的存放位置

    save_path = os.path.join(DEFAULT_SAVE_DIR, dataset)  # 将原始标签和图片从各个文件夹中复制出来后所存放的文件夹
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 将所有的图片和标签复制到一起
    cnt = 0
    whole_label_list = []
    for parent, dirnames, filenames in os.walk(original_label):
        for filename in filenames:
            if filename[-4:] == 'json':
                cnt += 1
                full_filename = os.path.join(parent, filename)
                name = filename[:-20]
                img_name = name + 'leftImg8bit.png'
                with open(full_filename, 'r') as f_handle:
                    data_json = json.load(f_handle)

                object_list = []

                for i in range(len(data_json['objects'])):
                    if data_json['objects'][i]['label'] in class_need:

                        class_name = data_json['objects'][i]['label']
                        s_l = data_json['objects'][i]['polygon']
                        x = []
                        y = []

                        for xy in s_l:
                            x.append(xy[0])
                            y.append(xy[1])

                        object_ = {
                            "class": str(std_class_ind[class_name]),
                            "label_id": int(class_label_id[class_name]),
                            "x1": int(min(x)),
                            "y1": int(min(y)),
                            "x2": int(max(x)),
                            "y2": int(max(y))
                        }
                        object_list.append(object_)

                json_str_from_txt = {
                    "img_name": img_name,
                    "path": full_filename,
                    "height": int(data_json['imgHeight']),
                    "width": int(data_json['imgWidth']),
                    "depth": 3,
                    "object": object_list
                }
                whole_label_list.append(json_str_from_txt)

    whole_label_dict = {"labels": whole_label_list}

    with open(os.path.join(save_path, "label.json"), 'w') as out_file:
        json.dump(whole_label_dict, out_file, ensure_ascii=False, indent=2)

    print('--------: ', cnt)


if __name__ == '__main__':
    print('-------------------------------')
    label_proc(dataset='val')
    print('===============================')
