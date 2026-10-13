# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 16:19:08 2017

@author: jlarsch
"""

import fnmatch
import os
import datetime
start='D:\\data\\b\\2017\\'
from models.experiment import experiment
import glob

matches = []
for root, dirnames, filenames in os.walk(start):
    for filename in fnmatch.filter(filenames, '*.avi'):
        try:
            datetime_object = datetime.datetime.strptime(filename[-18:-4], '%Y%m%d%H%M%S')
            if datetime_object > datetime.datetime(2017, 2, 6,23,0,0):
                matches.append(os.path.join(root, filename))
        except:
            pass
        

for p in matches:
    head, tail = os.path.split(p)
    txt_path=glob.glob(head+'\\Position*.txt')[0]
    print p
    experiment(p, txt_path)