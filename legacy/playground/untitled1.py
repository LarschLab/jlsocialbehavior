# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:34:36 2016

@author: jlarsch
"""

import numpy as np
import joFishHelper

class Animal(object):
    def __init__(self, id):
        self.id = id
        self.trajectory = np.asanyarray([])
        

class Pair(object):
    def __init__(self):
        self.animal_1 = Animal(1)
        self.animal_2 = Animal(2)
        self.distances =  np.asanyarray([])
        self.get_distances()        
        
    def get_distances(self):
        dist=self.animal_1.trajectory-self.animal_2.trajectory
        self.distances = np.sqrt(dist[:,0]**2 + dist[:,1]**2)
        

        
#experiment=joFishHelper.experiment(af)
#t=experiment.Pair.positionPx

pair = Pair()
pair.animal_1.trajectory=t[:,0,:]
pair.animal_2.trajectory=t[:,1,:]


