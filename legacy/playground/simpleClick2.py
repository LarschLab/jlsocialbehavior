# -*- coding: utf-8 -*-
"""
Created on Fri Dec 04 10:13:20 2015

@author: jlarsch
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

im=np.random.rand(10,10)
ax = plt.gca()
fig = plt.gcf()
implot = ax.imshow(im)

def onclick(event):
    if event.xdata != None and event.ydata != None:
        print(event.xdata, event.ydata)
        
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()