# -*- coding: utf-8 -*-
"""
Created on Fri Dec 04 14:16:45 2015

@author: jlarsch
"""
import subprocess as sp


aviP='C:/Users/jlarsch/Desktop/x264Test.avi'
 
# encoding speed:compression ratio
PRESET = 'fast'
 
# path to ffmpeg bin
#FFMPEG_PATH = 'c:/ffmpeg/bin/ffmpeg.exe'
FFMPEG_PATH = 'ffmpeg.exe'


sp.call('ffmpeg.exe -i C:/Users/jlarsch/Desktop/x264Test.avi -filter_complex "[0:v]crop=1024:1024:0:0[out1];[0:v]crop=1024:1024:1023:0[out2];[0:v]crop=1024:1024:0:1023[out3];[0:v]crop=1024:1024:1023:1023[out4]" -map [out1] out1.mp4 -map [out2] out2.mp4 -map [out3] out3.mp4 -map [out4] out4.mp4', shell=True)