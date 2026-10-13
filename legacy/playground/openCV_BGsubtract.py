# -*- coding: utf-8 -*-
"""
Created on Mon Dec 07 11:34:56 2015

@author: jlarsch
"""

import numpy as np
import cv2

avi_path = 'D:/data/b/2FishSeries_2/20151125_isolatedVsGroup/expStream2015-11-25T16_45_05_isolatedVsGroup.avi'
#cap = cv2.VideoCapture('C:/Users/jlarsch/Desktop/testVideo/x264Test.avi')
cap = cv2.VideoCapture(avi_path)
img1=cap.read()
gray = cv2.cvtColor(img1[1], cv2.COLOR_BGR2GRAY)
allMed=gray.copy()
fr=0

for i in range(10,900,100):
    cap.set(cv2.CAP_PROP_POS_FRAMES,i)
    image=cap.read()
    gray = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)
    
    allMed=np.dstack((allMed,gray))
    fr+=1
    
vidMed=np.median(allMed,axis=2)


height,width,layers=img1[1].shape
fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
videoOut1=cv2.VideoWriter('vid1.avi',fourcc,30,(900,900))
videoOut2=cv2.VideoWriter('vid2.avi',fourcc,30,(900,900))
videoOut3=cv2.VideoWriter('vid3.avi',fourcc,30,(900,900))
videoOut4=cv2.VideoWriter('vid4.avi',fourcc,30,(900,900))

fr=0


cap.set(cv2.CAP_PROP_POS_FRAMES,6)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

while(np.less(fr,30*60)):
    image=cap.read()
    gray = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)

    bgDiv=gray/vidMed
    cv2.imshow('Image',bgDiv)
    k = cv2.waitKey(1) & 0xff
    videoOut1.write(bgDiv[0:900,900:1800])
    videoOut2.write(bgDiv[900:1800,0:900])
    videoOut3.write(bgDiv[900:1800,900:1800])
    videoOut4.write(bgDiv[0:900,0:900])
    fr += 1
    if k == 27:
        break
 
cap.release()
cv2.destroyAllWindows()
videoOut1.release()
videoOut2.release()
videoOut3.release()
videoOut4.release()