# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 09:10:49 2017

@author: jlarsch
"""

import fnmatch
import os

head='D:\\data\\b\\2017\\20170118_VR_skypeVsFixed\\'

matches = []
for root, dirnames, filenames in os.walk(head):
    for filename in fnmatch.filter(filenames, 'PositionTxt_an*.txt'):
        matches.append(os.path.join(root, filename))
        
avimatches = []
for root, dirnames, filenames in os.walk(head):
    for filename in fnmatch.filter(filenames, '*.avi'):
        avimatches.append(os.path.join(root, filename))