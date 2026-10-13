# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:02:39 2016

@author: jlarsch
"""

import tkFileDialog
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
from tifffile import imsave

#select a directory containing tif-stacks to process
#tif-stacks must have equal dimensions

sd=tkFileDialog.askdirectory()

#list of files to process
fl= glob.glob(os.path.join(sd,'*.tif'))

#number of files to process
n_files=np.shape(fl)[0]

im = Image.open(fl[0])

#get number of frames in first file by seeking all frames until error occurs
#this seems clunky but is very fast
#assuming same number of frames for all files
n = 0
while True:
    n += 1
    try:
        im.seek(n)
    except:
        break
n_frames=n

#loop through all images,
#read each frame and accumulate frame-wise sum over all stacks

w, h = im.size
temp = np.zeros( (h,w,n_frames), dtype=np.int32 )

for i in range(n_files):
    print 'processing file: ', i
    im = Image.open(fl[i])
    
    for n in range (n_frames):
        curframe = np.array(im.getdata()).reshape(h,w)
        temp[:,:,n] += curframe
        im.seek(n)
        print ['frame: ', n],"         \r",

avgStack=temp/n_files
avgStack8b=avgStack.astype('int8')


