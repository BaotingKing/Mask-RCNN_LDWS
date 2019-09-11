#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: ZK
# Time: 2019/07/24 19:16
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import keras
from mrcnn import utils
import mrcnn.model as modellib
# from mrcnn import model
from mrcnn import visualize
from mrcnnBDD.LANE import BddDataset, BddConfig

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

print('[Step 1]: Load Data Set........................................................................................')
print('Training dataset....')
print('Validation dataset....')
dataset_train = BddDataset()
print('********: ', type(dataset_train), dataset_train)


print('[Step 2]: Load Model...........................................................................................')
# DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
# config = BddConfig()
# config.display()

# with tf.device(DEVICE):
#     model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
#                               config=config)
#
# # Load weights
# print("Loading weights ", weights_path)
# model.load_weights(weights_path, by_name=True)