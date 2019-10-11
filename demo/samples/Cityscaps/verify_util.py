#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: ZK
# Time: 2019/09/20 17:35
import os
import json
import cv2
import time
import random
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from demo.samples.Cityscaps.proc_cityscaps import search_file
from demo.samples.Cityscaps import utils_visual

def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    rgb = []
    for i in range(3):
        color = ""
        for j in range(2):
            color += colorArr[random.randint(0, 14)]
        rgb.append(int(color, 16))
    return tuple(rgb)


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


def ver_cityscape():
    ROOT_DIR = os.path.abspath('')  # Root directory of the project
    DEFAULT_SAVE_DIR = os.path.join(ROOT_DIR, "cityscapes")

    save_path = os.path.join(DEFAULT_SAVE_DIR, "train")
    with open(os.path.join(save_path, "label.json"), 'r') as f:
        infile = json.load(f)

    cnt = 0
    for img_info in infile['labels']:
        cnt += 1
        print(cnt, img_info['img_name'])
        dataset_path = 'G:\\Dataset\\Cityscape\\cityscaps\\leftImg8bit'
        img_path = search_file(dataset_path, img_info['img_name'])
        label_img_name = img_info['path'][:-20] + 'gtFine_labelIds.png'
        img_org = cv2.imread(img_path)
        img = img_org.copy()
        # cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Result", 1024, 512)
        objects = img_info['object']
        for obj in objects:
            points = []
            rng = randomcolor()
            segmentation = obj['segmentation'][0]
            bbox = obj['bbox']
            for i in range(int(len(segmentation) / 2)):
                points.append([segmentation[i * 2], segmentation[i * 2 + 1]])
            pts = np.array(points, np.int32)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), rng, 3)
            cv2.putText(img, obj['class'], (bbox[0], bbox[1]), cv2.FONT_HERSHEY_PLAIN, 2.0, rng, 2, 1)
            cv2.polylines(img, [pts], True, rng, 2)

            mask, class_ids = utils_visual.load_mask(obj, img_info['height'], img_info['width'])
            masked_image = utils_visual.apply_mask(img, mask[:, :, 0], rng)
        imgs = np.hstack([img_org, masked_image])
        cv2.imshow('Result', imgs)
        cv2.waitKey(0)


def ver_coco():
    COCO_PATH = "F:\\DataSet_0\\COCO\\"
    anno_path = "F:\\DataSet_0\\COCO\\coco2017\\train&val\\annotations"
    coco = COCO(os.path.join(anno_path, "instances_val2017.json"))
    for k_img_id, v in coco.imgs.items():
        img_path = search_file(COCO_PATH, v['file_name'])
        img_org = cv2.imread(img_path)
        img = img_org.copy()
        if k_img_id not in coco.imgToAnns.keys():
            continue
        objects = coco.imgToAnns[k_img_id]
        for obj in objects:
            points = []
            rng = randomcolor()
            #TODO **********************
            if type(obj['segmentation']) is dict:
                continue
            segmentation = obj['segmentation'][0]
            bbox = [int(i) for i in obj['bbox']]
            for i in range(int(len(segmentation) / 2)):
                points.append([segmentation[i * 2], segmentation[i * 2 + 1]])
            pts = np.array(points, np.int32)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), rng, 3)
            cv2.putText(img, str(obj['category_id']), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_PLAIN, 2.0, rng, 2, 1)
            cv2.polylines(img, [pts], True, rng, 2)
            mask, class_ids = utils_visual.load_mask(obj, img.shape[0], img.shape[1])
            masked_image = utils_visual.apply_mask(img, mask[:, :, 0], rng)
        imgs = np.hstack([img_org, masked_image])
        cv2.imshow('Result', imgs)
        cv2.waitKey(0)


if __name__ == '__main__':
    print('----------------------begin')
    # ver_coco()
    ver_cityscape()
    print('======================end')
