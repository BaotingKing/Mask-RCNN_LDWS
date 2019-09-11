#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: ZK
# Time: 2019/07/23 12:04
"""
    Mask R-CNN Demo
    A quick intro to using the pre-trained model to detect and segment objects.
"""
import os
import sys
import random
import time
import math
import numpy as np
import skimage.io

import PyQt5
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
# from mrcnn import model
from mrcnn import visualize
from demo import visualize2
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from samples.coco import coco
matplotlib.use('TkAgg')

# %matplotlib inline
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

print('***************************************************************************************************************')

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_DIR = 'F:\\DataSet_0\\COCO\\coco2017\\'
IMAGE_DIR = 'F:\\DataSet_0\\COCO\\coco2017\\train&val\\train2017\\'
IMAGE_DIR = 'F:\\DataSet_0\\BDD100k\\bdd100k_images\\bdd100k\\images\\100k\\train\\'


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # IMAGE_MIN_DIM = 720
    # IMAGE_MAX_DIM = 1280

config = InferenceConfig()
config.display()

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))


# Step1:  Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

# Step2: Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

cnt = 0
while True:
    image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))  # Load a random image from the images folder
    print('-----------------------------------------------------')
    # Step3: Run detection
    begin_time = time.time() * 1000
    results = model.detect([image], verbose=1)
    end_time = time.time() * 1000
    run_time = int(round(end_time - begin_time))
    print('begin_time = {0}/ms end_time = {1}/ms run_time = {2}/ms'.format(begin_time, end_time, run_time))

    # Visualize results
    r = results[0]
    if False:
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'])
    else:
        visualize2.cv_display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        class_names, r['scores'])

    cnt += 1
    if cnt == 20:
        break

