# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 09:47:34 2017

@author: jlarsch
"""

import pandas as pd
import tkFileDialog
import os
import numpy as np
from models.experiment import experiment

avi_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath('D:\\data\\b\\2017\\'))   
txt_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath('D:\\data\\b\\2017\\'))   

df=pd.read_csv(txt_path,header=None,delim_whitespace=True)
p1=df[[0,1,3,4]].values[:30*60*120].reshape(-1,2,2)
e=experiment(avi_path, txt_path, data=p1, forceCorrectPixelScaling=False, anSize=np.array([[5., 5.], [5, 5]]))
e.plotOverview()

p1=df[[6,7,9,10]].values[:30*60*120].reshape(-1,2,2)
e=experiment(avi_path, txt_path, data=p1, forceCorrectPixelScaling=False, anSize=np.array([[5., 5.], [5, 5]]))
e.plotOverview()

p1=df[[12,13,15,16]].values[:30*60*120].reshape(-1,2,2)
e=experiment(avi_path, txt_path, data=p1, forceCorrectPixelScaling=False, anSize=np.array([[5., 5.], [5, 5]]))
e.plotOverview()

p1=df[[0,1,9,10]].values[:30*60*120].reshape(-1,2,2)
e=experiment(avi_path, txt_path, data=p1, forceCorrectPixelScaling=False, anSize=np.array([[5., 5.], [5, 5]]))
e.plotOverview()