import matplotlib.pyplot as plt
import numpy as np
import cv2
import tkFileDialog
import wx
import os

import functions.getVideoProperties
import functions.video_functions as vf
import time
import pickle
#import ffmpegSplit4_module_bgDiv


avi_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath('c:/test/'))
#avi_path = 'C:/Users/jlarsch/Desktop/testVideo/x264Test.avi'
#avi_path = 'D:/data/b/2FishSeries_2/20151125_isolatedVsGroup/expStream2015-11-25T16_45_05_isolatedVsGroup.avi'

class UnTileArenaVideo(object):
    #split video recordings of multiple arenas into separate files
    def __init__(self,avi_path):
        
        self.avi_path=avi_path
        head, tail = os.path.split(self.avi_path)
        vp= functions.getVideoProperties.getVideoProperties(avi_path)
        self.ffmpeginfo = vp
        self.videoDims = [vp['width'] , vp['height']]
        self.numFrames=int(vp['nb_frames'])
        self.fps=vp['fps']


            
  
        
Scl=UnTileArenaVideo(avi_path)
