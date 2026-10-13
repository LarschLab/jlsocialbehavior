# -*- coding: utf-8 -*-
"""
Created on Fri Dec 04 14:16:45 2015

@author: jlarsch
"""
# split video into tiles based on tileList (width, height, x, y)
import subprocess as sp
import os
import numpy as np

def videoSplit(aviP,tileList):
    
    # path to ffmpeg bin
    #FFMPEG_PATH = 'c:/ffmpeg/bin/ffmpeg.exe'
    FFMPEG_PATH = 'ffmpeg.exe'
    
    head, tail = os.path.split(aviP)
    
    #assemble ffmpeg command
    
    fc=[] #start with empty filter command
    mc=[] #start with empty map command
    for i in range(np.shape(tileList)):    
        fcNew=['[0:v]crop=',tileList[i,0],':',tileList[i,1],':',tileList[i,2],':',tileList[i,3],':','[out1];']
        fc=fc+''.join(str(w) for w in fcNew)
        
        mcn1='[out'+str(i)+']'
        mcn2='out'+str(i)+'.avi'
        mcNew=['-map',mcn1,mcn2]
        mc=mc.e(mcNew)
    
    cmd=[FFMPEG_PATH,
    '-i', aviP,
    '-filter_complex', '[0:v]crop=1024:1024:0:0[out1];[0:v]crop=1024:1024:1023:0[out2];[0:v]crop=1024:1024:0:1023[out3];[0:v]crop=1024:1024:1023:1023[out4]',
    '-map', '[out1]', 'out1.mp4',
    '-map', '[out2]', 'out2.mp4',
    '-map', '[out3]', 'out3.mp4',
    '-map', '[out4]', 'out4.mp4']
    
    sp.check_output(cmd)