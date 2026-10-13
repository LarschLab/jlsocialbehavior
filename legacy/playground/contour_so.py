# -*- coding: utf-8 -*-
"""
Created on Fri Sep 09 13:56:08 2016

@author: jlarsch
"""

contours = [[[ 1,  1]],

   [[ 1, 36]],

   [[63, 36]],

   [[64, 35]],

   [[88, 35]],

   [[89, 34]],

   [[94, 34]],

   [[94,  1]]]
   
max_width = max(contours, key=lambda r: r[0] + r[2])[0]
max_height = max(contours, key=lambda r: r[3])[3]
nearest = max_height * 1.4
contours.sort(key=lambda r: (int(nearest * round(float(r[1])/nearest)) * max_width + r[0]))

for i in [1,4]:
    print i
    
    