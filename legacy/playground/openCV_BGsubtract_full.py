# -*- coding: utf-8 -*-
"""
Created on Mon Dec 07 11:34:56 2015

@author: jlarsch
"""

import numpy as np
import cv2

avi_path = 'E://b//2020//20210519_00_freeSwim_81cNTR//out_id0_30fps_20210519105550.avi'
#cap = cv2.VideoCapture('C:/Users/jlarsch/Desktop/testVideo/x264Test.avi')
cap = cv2.VideoCapture(avi_path)
img1=cap.read()
gray = cv2.cvtColor(img1[1], cv2.COLOR_BGR2GRAY)
allMed=gray.copy()
fr=0

for i in range(1000,90000,1000):
    cap.set(cv2.CAP_PROP_POS_FRAMES,i)
    image=cap.read()
    gray = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)
    
    allMed=np.dstack((allMed,gray))
    fr+=1
    
vidMed=np.median(allMed,axis=2)


height,width,layers=img1[1].shape
fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
videoOut1=cv2.VideoWriter('vid1.avi',fourcc,30,(width,height))

fr=0


cap.set(cv2.CAP_PROP_POS_FRAMES,30*30*60)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

fr = 0
while(np.less(fr,30*30*60)):
#while (np.less(fr, 300)):
    image = cap.read()
    gray = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)

    bgDiv=np.clip(255* (gray/vidMed),0,255)
    bgDiv = bgDiv.astype('uint8')
    bgDiv = cv2.cvtColor(bgDiv, cv2.COLOR_GRAY2BGR)
    if (fr%100) == 0:
        cv2.imshow('Image',bgDiv)
        print(fr)
    k = cv2.waitKey(1) & 0xff
    videoOut1.write(bgDiv)

    fr += 1
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
videoOut1.release()