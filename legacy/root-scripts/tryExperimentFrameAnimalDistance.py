# -*- coding: utf-8 -*-
"""
Created on Mon Feb 06 13:33:44 2017

@author: jlarsch
"""

import numpy as np
import functions.video_functions as vf
import matplotlib.pyplot as plt


def getAnimalSize(avi_path,experiment,needFrames=2000,numFrames=40000,boxSize=200):

    haveFrames=0
    frames=np.zeros(needFrames).astype('int')
    dist=np.zeros(needFrames)
    
    triedFr=[]
    triedD=[]
    while haveFrames<needFrames:
        tryFrame=np.random.randint(500,numFrames,1)
        minDist=np.max(np.abs(np.diff(experiment.rawTra[tryFrame,:,:],axis=1)))
        if minDist>boxSize:
            frames[haveFrames]=int(tryFrame)
            dist[haveFrames]=minDist
            haveFrames += 1
        else:
            triedFr.append(tryFrame)
            triedD.append(minDist)
    
    
    tra=experiment.rawTra[frames,:,:]
    tra[:,1,0]=tra[:,1,0]+512
    
    tmp=getAnimalLength(avi_path,frames,tra)
    
    MA=np.max(tmp[:,:,2:4],axis=2)
    bins=np.linspace(0,100,101)
    anSize=[np.argmax(np.histogram(MA[:,0],bins=bins)[0]),np.argmax(np.histogram(MA[:,1],bins=bins)[0])]
    
    return anSize

#plt.figure()
##plt.hist(MA)
#plt.hist(MA,bins=bins)
#
#plt.figure()
#plt.scatter(MA[:,0],tmp[:,0,4],color='b')
#plt.scatter(MA[:,1],tmp[:,0,4],color='g')