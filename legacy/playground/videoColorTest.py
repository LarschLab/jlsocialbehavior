# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 11:10:58 2016

@author: jlarsch
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def processing_video(path,frames,start=0):
    
    #read video frame
    outSize=200
    video = cv2.VideoCapture(path)
    img_frame_first = video.read()[1]
    height_ori, width_ori = img_frame_first.shape[:2]
    
    frameList=np.arange(start,frames,1)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videoOut1=cv2.VideoWriter('d:/vid1.avi',fourcc,30,(outSize,outSize),True)
    
    
    
    for i in frameList:
        video.set(cv2.CAP_PROP_POS_FRAMES,i)
        img_frame_original = video.read()[1]
        img=cv2.resize(img_frame_original,(outSize,outSize))
        cv2.circle(img,(100,100),5,(255,0,0),1)
        videoOut1.write(img)
        plt.imshow(img) #this shows a red circle
    
    videoOut1.release()