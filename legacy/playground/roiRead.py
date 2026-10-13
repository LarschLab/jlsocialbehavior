# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 18:01:32 2017

@author: jlarsch
"""

import tkFileDialog
import numpy as np

roiPath = tkFileDialog.askopenfilename()
rois=np.loadtxt(roiPath)
r_px=rois.mean(axis=0)[-1]