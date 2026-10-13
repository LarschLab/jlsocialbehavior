# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 00:11:46 2016

@author: johannes
"""

import cv2
import numpy as np
import matplotlib
#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

img = np.ones((50, 50), np.uint8)
cv2.imshow('image', img)
plt.figure()
plt.plot([1,2], [1,3])
plt.show()