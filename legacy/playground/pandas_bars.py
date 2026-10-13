# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 22:43:34 2016

@author: johannes
"""

import pandas as pd

df = pd.DataFrame({'a':[10, 20], 'b': [15, 25], 'c': [35, 40], 'd':[45, 50]}, index=['john', 'bob'])

ax = df[['a', 'c']].plot.bar(position=0, width=0.1, stacked=True)
df[['b', 'd']].plot.bar(position=1, width=0.1, stacked=True, ax=ax)

ax=df[['b', 'd']].plot.bar(position=0, width=0.1, stacked=True)
df[['a', 'c']].plot.bar(position=1, width=0.1, stacked=True, ax=ax)
