# -*- coding: utf-8 -*-
"""
Created on Tue Dec 01 11:58:49 2015

@author: jlarsch
"""

# import the necessary packages
import numpy as np
#import argparse
import cv2
import tkFileDialog
import os


 
# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required = True, help = "Path to the image")
#args = vars(ap.parse_args())

avi_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b'))

capture = cv2.VideoCapture(avi_path)
capture.set(cv2.CAP_PROP_POS_FRAMES,100)


# load the image, clone it for output, and then convert it to grayscale
image = capture.read()
output = image[1].copy()
gray = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)


# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 500)
 
# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
     
    	# loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
    		# draw the circle in the output image, then draw a rectangle
    		# corresponding to the center of the circle
    	cv2.circle(output, (x, y), r, (0, 255, 0), 4)
    	cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
     
    	# show the output image
    small = cv2.resize(output, (0,0), fx=0.25, fy=0.25) 
    cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
    cv2.imshow("output", output)
    cv2.waitKey(0)