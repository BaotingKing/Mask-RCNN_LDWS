#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: ZK
# Time: 2019/09/19 19:20
import os
import json
import cv2
import shutil

class_need = ['person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']

std_class_ind = {'car': 'car',
                 'truck': 'truck',
                 'caravan': 'truck',
                 'trailer': 'truck',
                 'train': 'train',
                 'bicycle': 'bike',
                 'person': 'person',
                 'rider': 'rider',
                 'motorcycle': 'bike',
                 'bus': 'bus',
                 }

original_img = 'G:\\Dataset\\Cityscape\\cityscapes\\leftImg8bit\\val'  # 原始图片的存放位置
original_label = 'G:\\Dataset\\Cityscape\\cityscaps_label\\gtFine\\val'  # 原始标签的存放位置

ROOT_DIR = os.path.abspath('')  # Root directory of the project
DEFAULT_SAVE_DIR = os.path.join(ROOT_DIR, "cityscapes")

save_path = os.path.join(DEFAULT_SAVE_DIR, "val")  # 将原始标签和图片从各个文件夹中复制出来后所存放的文件夹
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 将所有的图片和标签复制到一起
for parent, dirnames, filenames in os.walk(original_label):
    for filename in filenames:
        if filename[-4:] == 'json':
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
                        "x1": int(min(x)),
                        "y1": int(min(y)),
                        "x2": int(max(x)),
                        "y2": int(max(y))
                    }
                    object_list.append(object_)

            json_str_from_txt = {
                "img_name": img_name,
                "height": int(data_json['imgHeight']),
                "width": int(data_json['imgWidth']),
                "depth": 3,

                "object": object_list
            }
            json_str = json.dumps(json_str_from_txt, indent=2)
            with open(save_path + name + 'leftImg8bit.json', 'w') as f:
                f.write(json_str)
