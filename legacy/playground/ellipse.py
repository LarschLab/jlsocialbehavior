# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:54:18 2016

@author: jlarsch
"""

import numpy as np
import matplotlib.pyplot as plt
import math
width = 128
height = 128

dtheta=2*math.pi/width
theta=np.arange(0,2*math.pi,dtheta)
x=2*np.cos(theta)
y=np.sin(theta)

plt.plot(x,y,'o-')
plt.axes().set_aspect('equal')