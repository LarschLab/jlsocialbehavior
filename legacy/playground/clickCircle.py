
import matplotlib.pyplot as plt
import numpy as np
#import argparse
import cv2
import tkFileDialog
import os


avi_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b'))

capture = cv2.VideoCapture(avi_path)
capture.set(cv2.CAP_PROP_POS_FRAMES,100)



# load the image, clone it for output, and then convert it to grayscale
image = capture.read()
output = image[1].copy()
gray = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)

class MyClickableImage(object):
    def __init__(self,frame):
        

        self.frame = frame
        self.fig = plt.figure(figsize=(6,9))
        self.ax = self.fig.add_subplot(111)
        xaxis = self.frame.shape[1]
        yaxis = self.frame.shape[0]
        self.im = self.ax.imshow(self.frame[::-1,:], 
                  cmap='jet', extent=(0,xaxis,0,yaxis), 
                  picker=5)
        line, = self.ax.plot([0], [0])  # empty line
        self.x = list(line.get_xdata())
        self.y = list(line.get_ydata())
        self.line = line
        self.fig.canvas.mpl_connect('button_press_event', self.onpick1)
        #button_press_event
        plt.show()
        self.ClickCount=0

    # some other associated methods go here...

    def onpick1(self,event):
        
        self.x.append(event.xdata)
        self.y.append(event.ydata)
        self.line.set_data(self.x, self.y)
        self.line.figure.canvas.draw()
        
        #self.y = event.ydata
            
cl=MyClickableImage(gray)