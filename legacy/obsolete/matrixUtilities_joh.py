# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 20:52:49 2015

@author: johannes
"""
import numpy as np

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return np.array([theta, rho])
    
def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return np.array([x, y])
    
def runningMean(x, N):
    return np.convolve(x, np.ones((N,))/N,mode='valid')[(N-1):]