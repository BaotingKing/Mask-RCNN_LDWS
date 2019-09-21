#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: ZK
# Time: 2019/09/20 17:35
import os
import json
import cv2
import time
import numpy as np

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

save_path = os.path.join(DEFAULT_SAVE_DIR, "val")
with open(os.path.join(save_path, "label.json"), 'r') as f:
    infile = json.load(f)

cnt = 0
for img_info in infile['labels']:
    print(cnt, img_info['img_name'])
    cnt += 1
    label_img_name = img_info['path'][:-20] + 'gtFine_labelIds.png'
    img = cv2.imread(label_img_name)
    if img.shape[0] != img_info["height"] or img.shape[1] != img_info["width"]:
        m = np.ones([img_info["height"], img_info["width"]], dtype=bool)

    m = np.ones([img_info["height"], img_info["width"]], dtype=bool)

    for obj in img_info['object']:
        m = np.ones([img_info["height"], img_info["width"]], dtype=bool)
        mask = img[:, :, 0]
        mask_list = mask.flatten()
        pingpong = 0
        rle = []
        start = time.time()

        idx = list(np.where(mask_list == obj['label_id'])[0])
        temp = np.ones(idx == obj['label_id'])

        for i in idx:
            if len(rle) == 0:
                rle.append(int(i) + 1)
                pingpong = i
                cnt = 0
                continue

            if abs(i - pingpong) == 1:
                cnt += 1
                pingpong = i
                continue
            else:
                rle.append(cnt + 1)
                cnt = 0
            pingpong = i
        suma = sum(rle)
        total = time.time() - start
        cv2.imshow('lala', mask)
        cv2.waitKey(0)

