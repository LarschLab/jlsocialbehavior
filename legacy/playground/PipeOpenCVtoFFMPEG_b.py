# -*- coding: utf-8 -*-
"""
Created on Wed Dec 09 13:30:29 2015

@author: jlarsch
"""

# This script copies the video frame by frame
import cv2
import subprocess as sp


input_file = 'C:/Users/jlarsch/Desktop/testVideo/x264Test.avi'
output_file = 'C:/Users/jlarsch/Desktop/testVideo/b/output_file_name.mp4'

cap = cv2.VideoCapture(input_file)
ret, frame = cap.read()
height, width, ch = frame.shape

ffmpeg = 'ffmpeg.exe'
dimension = '{}x{}'.format(width, height)
f_format = 'bgr24' # remember OpenCV uses bgr format
fps = str(cap.get(cv2.CAP_PROP_FPS))

command = [ffmpeg,
        '-y',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', dimension,
        '-pix_fmt', 'bgr24',
        '-r', fps,
        '-i', '-',
        '-an',
        '-vcodec', 'mpeg4',
        '-b:v', '5000k',
        output_file ]

proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    proc.stdin.write(frame.tostring())

cap.release()
proc.stdin.close()
proc.stderr.close()
proc.wait()