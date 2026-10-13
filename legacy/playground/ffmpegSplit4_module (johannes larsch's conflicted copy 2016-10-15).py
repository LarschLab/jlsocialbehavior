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
    
    fc='' #start with empty filter command
    mc=[''] #start with empty map command
    for i in range(np.shape(tileList)[0]):
        
        #create subdirectories for split output
        directory=head+'/'+ str(i+1)+'/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        fcNew=('[0:v]crop={0:.0f}:{1:.0f}:{2:.0f}:{3:.0f}:[out{4}];').format(*(tileList[i]+[i+1])) #e.g. [0:v]crop=1024:1024:0:0[out1];
        fc=fc+''.join(str(w) for w in fcNew)
        mcn1='[out'+str(i+1)+']'
        mcn2=directory+'split_'+str(i+1)+'_'+tail+'.mp4'
        mcNew=['-map',mcn1,mcn2]
        mc.extend(mcNew)

    cmd=[FFMPEG_PATH,
    '-i', aviP,
    '-filter_complex', fc[:-1]]

    cmd.extend(mc[1:])    
    print 'starting ffmpeg for video processing...'
    sp.check_output(cmd)
    

