# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:37:20 2016

@author: jlarsch
"""

import cv2
import numpy as np

def video_processing(path):
    fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    fps=30
    videoOut = cv2.VideoWriter(path,fourcc,fps,(100 ,100))

    #some processing:
    for i in range(10):
        result=np.ones((100,100))
        
    videoOut.write(result)

    
path='d:/zzvidTest.avi'
video_processing(path)