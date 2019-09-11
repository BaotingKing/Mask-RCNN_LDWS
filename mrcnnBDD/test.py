#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: ZK
# Time: 2019/07/12 17:42

import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon


def sample():
    pass


mask = np.ones([300, 300, 6], dtype=np.uint8)
mask = np.stack(mask, axis=2).astype(np.bool)
if 0:
    print('----------------:', mask)

# for i, (shape, _, dims) in enumerate(info['shapes']):
#     mask[:, :, i:i + 1] = self.draw_shape(mask[:, :, i:i + 1].copy(),
#                                           shape, dims, 1)
# # Handle occlusions
# occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
# for i in range(count - 2, -1, -1):
#     mask[:, :, i] = mask[:, :, i] * occlusion
#     occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
# # Map class names to class IDs.
# class_ids = np.array([self.class_names.index(s[0]) for s in shapes])


