# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:41:18 2019

@author: jlarsch
"""
import os

import functions.getMedVideo

codeDir = 'C:\\Users\\jlarsch\\Documents\\jlsocialbehavior'
os.chdir(codeDir)
import functions.paperFigureProps as pfp
import functions.gui_circle as gc
import functions.video_functions as vf

import matplotlib.pyplot as plt

base="E:\\b\\inbalOMR\\out_id0_30fps_20220316160019\\"
video = 'out_id0_30fps_20220316160019'
input_vidpath = base + video + '.avi'

vidMed,vidMed_fn= functions.getMedVideo.getMedVideo(input_vidpath)
fig, ax = plt.subplots(figsize=(20,20))
ax.imshow(vidMed)

rois,roi_fn=gc.get_circle_rois(vidMed_fn)
