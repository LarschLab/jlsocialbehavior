# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 09:50:51 2017

@author: jlarsch
"""

import numpy as np

a=np.ones((5,5))
np.fill_diagonal(a,0)

a[0]


x=25.
h=70.

alpha=np.arctan(x/h)
np.degrees(alpha)

h2=h-1
x2=np.tan(alpha)*h2
x2

import pandas as pd
