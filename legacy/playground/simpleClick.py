# -*- coding: utf-8 -*-
"""
Created on Fri Dec 04 08:54:13 2015

@author: jlarsch
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np


#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(np.random.rand(10))

class ClickImage(object):
    def __init__(self,frame):
        self.x = None
        self.y = None
        #self.im = self.ax.plot(np.random.rand(10))
        self.fig = plt.figure(figsize=(6,9))
        self.ax = self.fig.add_subplot(111)
        self.im = self.ax.plot(np.random.rand(10,10))
        self.fig.canvas.mpl_connect('motion_notify_event', self.onclick)
        plt.show()
        
    def onclick(self,event):
        if event.xdata != None and event.ydata != None:
            event.canvas.figure.clear()
            event.canvas.figure.gca().plot(np.random.rand(10,10))
            event.canvas.draw()
        
cid = ClickImage(np.random.rand(10,10))