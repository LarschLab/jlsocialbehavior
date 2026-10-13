# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 00:25:13 2016

@author: johannes
"""
import numpy as np


a=np.arange(5)
print 'a original', a
b=a
print 'b = a', b
a[0]=10
print 'a modified', a
print 'b ', b


import numpy as np
a=np.arange(5)
print 'a original', a
b=np.diff(a)
print 'b = np.diff(a)', b
a[0]=10

print 'a modified', a
print 'b ', b

def c(a):
    return np.diff(a)
    
import numpy as np
class linkedVariables:
    def __init__(self,a=np.arange(5)):
        self.a=a
        self.b=np.diff(self.a)
        
    def c(self):
        return np.diff(self.a)

lv=linkedVariables()

print 'a original', lv.a
print 'b = np.diff(a)', lv.b
print 'c = np.diff(a)', lv.c()
lv.a[0]=10

print 'a modified', lv.a
print 'b ', lv.b
print 'c ', lv.c()