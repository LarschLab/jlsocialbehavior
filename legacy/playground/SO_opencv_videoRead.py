# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 11:30:48 2016

@author: jlarsch
"""
import numpy as np
import cv2

#access a video from your disk
cap = cv2.VideoCapture('d:/data/eg_videoRead.avi')


#we are going to read 10 frames 
#we store the frames in a numpy structure
#then we'll generate a minimum projection of those frames

frameStack=[]
numFrames=10

for fr in range(numFrames):
    cap.set(cv2.CAP_PROP_POS_FRAMES,fr) #specifies which frame to read next
    frame=cap.read() #read the frame
    #gray = cv2.cvtColor(frame[1], cv2.COLOR_BGR2GRAY) #convert to gray scale
    frameStack.append(frame[1]) #add current frame to our frame Stack
    
minProjection=np.min(frameStack,axis=0) #find the minimum across frames
cv2.imshow("projection", minProjection) #show the result