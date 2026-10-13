# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 09:20:42 2017

@author: jlarsch
"""

import numpy as np
#import scipy.io
import tkFileDialog
import os

p = tkFileDialog.askopenfilename(initialdir=os.path.normpath('D:\\data\\b\\2017\\'))    


with open(p) as f:
    mat=np.loadtxt((x.replace(b'(',b' ').replace(b')',b' ') for x in f),delimiter=',')
    #mat[:,:-1].reshape((mat.shape[0],2,2)),[1,1]