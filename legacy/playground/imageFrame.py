# -*- coding: utf-8 -*-
"""
Created on Mon Aug 08 22:46:20 2016

@author: johannes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


im='d:/frame.png'
img=plt.imread(im)
fig, ax = plt.subplots()

frame_height=25
x_start=20
y_start=15
ax.imshow(img,extent=[x_start,x_start+frame_height,y_start,y_start+frame_height])

ax.add_patch(
        Polygon(
            [[0,0], [20, 15], [20, 40]],
            closed=True, fill=False, lw=1)
        )
ax.set_xlim(0, 60)
ax.set_ylim(0, 40)
plt.show()


a=np.random.random((100,5))
fig, ax = plt.subplots()
lines=ax.plot(a)
line_colors=[l.get_color() for l in lines]
