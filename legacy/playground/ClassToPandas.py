# -*- coding: utf-8 -*-
"""
Created on Thu Dec 03 11:47:17 2015

@author: jlarsch
"""


class foo(object):
    def __init__(self,input):
        self.val=input

allFoo=[];
for i in range(10):
    allFoo.append(foo(i))

