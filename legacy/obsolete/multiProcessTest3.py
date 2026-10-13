# -*- coding: utf-8 -*-
"""
Created on Wed Jul 06 22:02:29 2016

@author: jlarsch
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 11:23:04 2016

@author: jlarsch
"""

import multiprocessing as mp
from multiprocessing import Process, Queue
import numpy as np
import cv2
import tkFileDialog
import os

def main():
    path=tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b/2016/20160314_TLStacks/1/a/1/'))

    coordinates=np.random.random((1000,2))
    listOf_FuncAndArgLists=[]
    
    for i in range(2):
        frame_coordinates=coordinates[i,:]
        listOf_FuncAndArgLists.append([parallel_function,frame_coordinates,i,path])
        
    queues=[Queue() for fff in listOf_FuncAndArgLists] #create a queue object for each function
    jobs = [Process(target=storeOutputFFF,args=[funcArgs[0],funcArgs[1:],queues[iii]]) for iii,funcArgs in enumerate(listOf_FuncAndArgLists)]
    for job in jobs: job.start() # Launch them all
    #for job in jobs: job.join() # Wait for them all to finish
    # And now, collect all the outputs:
  
    return([queue.get() for queue in queues])         

def storeOutputFFF(fff,theArgs,que): #add a argument to function for assigning a queue
    print 'MULTIPROCESSING: Launching %s in parallel '%fff.func_name
    que.put(fff(*theArgs)) #we're putting return value into queue
      
def parallel_function(frame_coordinates,i,path):
    video = cv2.VideoCapture(path)
    a=[]
    b=np.zeros(5000)
    for i in range(5):
        video.set(cv2.CAP_PROP_POS_FRAMES,i)
        img_frame_original = video.read()[1]
        a.append(img_frame_original[1,1])
        b[1]=img_frame_original[1,1][0]
    return a
    
class resultClass(object):
    def __init__(self,maxIntensity,i):
        self.maxIntensity=maxIntensity
        self.i=i
    
if __name__ == '__main__':
    mp.freeze_support()
    a=main()