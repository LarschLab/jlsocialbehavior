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
FFMPEG_PATH = 'ffmpeg'


sp.call([FFMPEG_PATH,
'-r', '24',
'-i', aviP,
'-c:v', 'libx264',
'-preset', 'fast',
' output.mp4'])
 
