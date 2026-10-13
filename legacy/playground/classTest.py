# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 09:47:56 2016

@author: jlarsch
"""
import numpy as np

class Animal(object):
    def __init__(self):
        self.a=Trajectory(np.zeros((10,2)))
        self.b=Trajectory(np.ones((10,2)))

class Trajectory(object):
    def __init__(self,xy=np.array([])):
        self.xy=xy
        
    def x(self):
        return self.xy[:,0]
    
    def y(self):
        return self.xy[:,1]
        
class Pair(object):
    def __init__(self):
        self.animals=[]
        self.animals.append(Animal())
        self.animals.append(Animal())
        

    def get_var_from_all(self, var):
        tmp=self.animals
        for i in range(len(var)):
            tmp=[getattr(x, var[i]) for x in tmp]
            
        return np.stack(tmp,axis=-1)
        
pair=Pair()
pair.get_var_from_all(['x','a'])