#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: ZK
# Time: 2019/09/29 17:06
"""
    Mask R-CNN Demo
    A quick intro to using the pre-trained model to detect and segment objects.
"""
import os
import sys
import random
import time
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
# from mrcnn import model
# from mrcnn import visualize
from demo.show_info import visualize2
from demo.samples.Cityscaps.cityscapes import CityScapesConfig
from demo.samples.coco.coco_pro import CocoConfig

# Local path to trained weights file
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# NN_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# NN_MODEL_PATH = "F:\\projects\\Mask_RCNN\\mask_rcnn_coco.h5"
NN_MODEL_PATH = "F:\\projects\\Mask-RCNN_LDWS\\logs\\coco20191003T1719\\mask_rcnn_coco_0002.h5"


# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_DIR = 'F:\\DataSet_0\\BDD100k\\bdd100k_images\\bdd100k\\images\\100k\\train\\'
IMAGE_DIR = 'F:\\DataSet_0\\COCO\\coco2017\\test\\test2017\\test2017\\'

if False:
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
else:
    # class_names = ['BG', 'car', 'bus', 'bicycle', 'person', 'truck']
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'boat', 'traffic light', 'stop sign']


class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(class_names[1:])  # DATA set has classes num


config = InferenceConfig()
config.display()

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))


# Step1:  Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=NN_MODEL_PATH)

# Step2: Load weights
model.load_weights(NN_MODEL_PATH, by_name=True)

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
    if cnt == 1000:
        break

