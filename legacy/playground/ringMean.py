# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:33:38 2016

@author: jlarsch
"""

import numpy as np

#generate a random 300x300 matrix for testing
inputMat = np.random.random((300,300))
radius=50

def radMask(index,radius,array):
  a,b = index
  nx,ny = array.shape
  y,x = np.ogrid[-a:nx-a,-b:ny-b]
  mask = x*x + y*y <= radius*radius

  return mask
  

#meanAll is going to store ~18000 points
meanAll=np.zeros((130,130))

for x in range(130):
    for y in range(130):
        centerMask=(x,y)
        mask=radMask(centerMask,radius,inputMat)
        #un-mask center and values below 0
        mask[centerMask]=False
        mask[inputMat<0]=False
        
        #get the mean
        meanAll[x,y]=np.mean(inputMat[mask])

