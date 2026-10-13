# -*- coding: utf-8 -*-
"""
Created on Wed Dec 09 11:15:45 2015

@author: jlarsch
"""

import cv2
import subprocess as sp
import numpy as np
import time
import joFishHelper

#input_file = 'C:/Users/jlarsch/Desktop/testVideo/x264Test.avi'

input_file = 'C:/Users/jlarsch/Desktop/testVideo/b/test.avi'
#output_file=['C:/Users/jlarsch/Desktop/testVideo/b/output_file_name1.avi',
#'C:/Users/jlarsch/Desktop/testVideo/b/output_file_name2.avi',
#'C:/Users/jlarsch/Desktop/testVideo/b/output_file_name3.avi',
#'C:/Users/jlarsch/Desktop/testVideo/b/output_file_name4.avi']

output_file=['C:/Users/jlarsch/Desktop/testVideo/output_file_name1.avi',
'C:/Users/jlarsch/Desktop/testVideo/output_file_name2.avi',
'C:/Users/jlarsch/Desktop/testVideo/output_file_name3.avi',
'C:/Users/jlarsch/Desktop/testVideo/output_file_name4.avi']

cap = cv2.VideoCapture(input_file)
cap.set(cv2.CAP_PROP_POS_FRAMES,6)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
height, width, ch = frame.shape

ffmpeg = 'ffmpeg.exe'
dimension = '{}x{}'.format(width, height)
#dimension = '{}x{}'.format(long(1024), long(1024))
#format = 'bgr24' # remember OpenCV uses bgr format
fps = str(cap.get(cv2.CAP_PROP_FPS))

rois=[[0,1024,0,1024],
[1024,2048,0,1024],
[0,1024,1024,2048],
[1024,2048,1024,2048]]

vp=joFishHelper.getVideoProperties(input_file)
numFrames=int(vp['nb_frames'])
allMed=gray.copy()
for i in range(10,numFrames-2,np.round(numFrames/9)): #use 9 images to calculate median
    cap.set(cv2.CAP_PROP_POS_FRAMES,i)
 #   print i
    image=cap.read()
    gray = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)  
    allMed=np.dstack((allMed,gray))
    
vidMed=np.median(allMed,axis=2)
        
command =[]
proc=[]
for i in range(4):
    command.append([ ffmpeg, 
        '-y', 
        '-f', 'rawvideo', 
        '-vcodec','rawvideo',
        '-s', dimension,
        '-pix_fmt', 'gray',
        '-r', fps,
        '-i', '-',
        '-an',
        '-vcodec', 'mpeg4',
        '-b:v', '2000k',
        output_file[i] ])
    
    #proc.append(sp.Popen(command[i], stdin=sp.PIPE, stderr=sp.PIPE))
    proc.append(sp.Popen(command[i], stdin=sp.PIPE))
fr=0
t1=time.time()
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #bgDiv=gray/vidMed
    bgDiv=np.divide(gray,vidMed)
    bgDivI = cv2.convertScaleAbs(bgDiv,alpha=225,beta=20)
    fr+=1
    print fr
    #if not ret:
    if np.greater(fr,1000) or not ret:
        break
    
    for i in range(4):
        #frsm=bgDivI[rois[i][0]:rois[i][1],rois[i][2]:rois[i][3]]
        proc[i].stdin.write(bgDivI.tostring())

cap.release()
t2=time.time()
print 'this took this much time:', t2-t1
        
for i in range(4):
    proc[i].stdin.close()
    #proc[i].stderr.close()
    #proc[i].wait()