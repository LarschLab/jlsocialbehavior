# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 09:21:14 2016

@author: jlarsch
"""

import tkFileDialog
import joFishHelper
import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import cv2

avi_path = tkFileDialog.askopenfilename(initialdir='d:/data/b/2016/')
cap = cv2.VideoCapture(avi_path)
FramesToProcess=range(60*60*0,60*60*35)
nFramesToProcess=len(FramesToProcess)
sdPerFrame=np.zeros(nFramesToProcess)
img=np.zeros([1968,1968,2])
for i in range(nFramesToProcess):
    f=FramesToProcess[i]
    cap.set(cv2.CAP_PROP_POS_FRAMES,f)
    currImg=cap.read()[1]
    img[:,:,np.mod(i,2)]=cv2.cvtColor(currImg, cv2.COLOR_BGR2GRAY)
    sdPerFrame[i]=np.sum(np.abs(np.diff(img)))
    print f,"         \r",

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    
b=sdPerFrame[np.ix_(sdPerFrame<2000000)]
c=moving_average(b,60)
plt.plot(c[1:])