__author__ = 'jlarsch'

#import Tkinter
import tkFileDialog
import models.experiment as xp
import models.experiment_set as es
import os
import numpy as np

batchmode=1
timeStim=1
sizePlot=0
systShift=0
postureAnalysis=0
forceCorrectPixelScaling=0
episodes=-1
if batchmode:
    
    expSet=es.experiment_set(csvFile=[], systShift=systShift, timeStim=timeStim, sizePlot=sizePlot, episodes=episodes)
            
else:    
    #avi_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath('D:/data/b/2016/20160311_arenaSize/b/1/'))
    avi_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath('D:\\data\\b\\2017\\'))    
    #avi_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b/2016/20160314_TLStacks/1/a/1'))
    #rng=np.arange(10*60*30,50*30*60,1)
    experiment=xp.experiment(avi_path)
    experiment.plotOverview()


    
    