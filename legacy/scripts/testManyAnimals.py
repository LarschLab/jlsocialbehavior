# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 09:47:34 2017

@author: jlarsch
"""

import pandas as pd
import tkFileDialog
import os
from models.experiment import experiment

avi_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath('D:\\data\\b\\2017\\'))   
df=pd.read_csv(avi_path,header=None,delim_whitespace=True)
p1=df[[range(4)]]
e=experiment(avi_path,avi_path,data=p1)