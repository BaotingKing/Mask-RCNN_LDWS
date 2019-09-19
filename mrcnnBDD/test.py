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


def gen():
    i = 0
    while i < 5:
        temp = yield i
        print('=====temp', temp)
        print('=====i', i)
        i += 1


f = gen()
print('-----------================', f)
a = next(f)
print('-----------================0', a)
a = next(f)
print('-----------================1', a)
a = next(f)
print('-----------================2', a)
a = f.send('6666666666666666666666')
print('-----------================3', a)
a = next(f)
print('-----------================4', a)
