# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:24:41 2016

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
FramesToProcess=range(60*30*35)
nFramesToProcess=60*30*35
sdPerFrame=np.zeros(nFramesToProcess)
img=np.zeros([1968,1968,1])


#grayi=
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 230;
params.maxThreshold = 255;

#params.blobColor = 0
 
# Filter by Area.
params.filterByArea = True
params.minArea = 50
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False


# Set up the detector with  parameters.
detector = cv2.SimpleBlobDetector_create(params)
cv2.namedWindow('blobs',cv2.WINDOW_NORMAL)

keyAll=[]

for i in range(nFramesToProcess):
    img=cap.read()[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect blobs.
    keypoints = detector.detect(gray)
    keyAll.append(keypoints)
    
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
    # Show keypoints
    cv2.imshow("blobs", im_with_keypoints)
    cv2.waitKey(1)
    
allAllX=[]
allAllY=[]
    
for i in range(nFramesToProcess):
    allX=[]
    allY=[]
    allX.append([x.pt[0] for x in keyAll[i]])
    allY.append([x.pt[1] for x in keyAll[i]])
    allAllX.append(allX)
    allAllY.append(allY)
    
tAll=np.zeros([nFramesToProcess,150,2])
ts=np.squeeze(np.asarray([allAllX[0],allAllY[0]]))
tAll[0,0:np.shape(ts)[1],:]=ts.T

def closest_node(node, nodes):
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)
    
for i in range(nFramesToProcess-1):
    tnOld=np.squeeze(np.asarray([allAllX[i+1],allAllY[i+1]]))
    tc=(tAll[i,0:np.shape(tnOld)[1],:]).T
    tn=np.squeeze(np.asarray([allAllX[i+1],allAllY[i+1]]))
    for j in range(np.shape(tc)[1]):
        a=tc[:,j]
        cn=closest_node(a,tn.T)
        tAll[i+1,j,:]=tn[:,cn]

tAll_d=np.diff(tAll,axis=0)
travel=np.sqrt(tAll_d[:,:,0]**2 + tAll_d[:,:,1]**2)
travelb=travel.copy()
travelb[np.where(travel > 10)]=np.nan
plt.plot(np.nanmean(travelb,axis=1))

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    
    
c=moving_average(np.nanmean(travelb,axis=1),60)