# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:17:21 2017

@author: jlarsch
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkFileDialog
import os

csvPath = tkFileDialog.askopenfilename(initialdir='E:/00_bonsai_ffmpeg_out/')

raw=np.loadtxt(csvPath)
p=np.array([raw[::4],raw[1::4]]).T
d=np.array([raw[2::4],raw[3::4]]).T
h, status = cv2.findHomography(d, p)
h.tolist()
#head, tail = os.path.split(csvPath)

#np.savetxt()

#img1=np.zeros((2000,2000),dtype='int8')

#for x in p:
#    cv2.circle(img1,(x[0],x[1]),20,(250,250,250),-1)
    
#img2=np.zeros((2000,2000),dtype='int8')    
#for x in d:
#    cv2.circle(img1,(x[0],x[1]),20,(50,250,0),-1)
    



#h2 = cv2.getPerspectiveTransform(np.array([d.astype('float32')]), np.array([p.astype('float32')]))
#h2[h2<0.01]=0

#im_dst = cv2.warpPerspective(img2, h, (2000,2000))



#dc=cv2.perspectiveTransform(np.array([d.astype('float32')]), h)[0]
#for x in dc:
#    cv2.circle(img1,(x[0],x[1]),5,(100,250,0),-1)
#    plt.figure('img1')
#plt.imshow(img1)