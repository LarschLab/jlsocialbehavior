# -*- coding: utf-8 -*-
"""
Created on Mon Dec 07 11:34:56 2015

@author: jlarsch
"""

import numpy as np
import cv2
import os




def videoSplit(avi_path,tileList):
    
    head, tail = os.path.split(avi_path)
    VidOutList=[]
    
    for i in range(np.shape(tileList)[0]):
    #for i in range(1):
    #i=0
        #create subdirectories for split output
        directory=head+'/'+ str(i+1)+'/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        VidOutFile=(directory+'split_'+str(i+1)+'_'+tail)
        print VidOutFile
        fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        VidOutList.append(cv2.VideoWriter(VidOutFile,fourcc,30,(np.int(tileList[i][0]),np.int(tileList[i][1]))))
       # VidOutList.append(cv2.VideoWriter(VidOutFile,fourcc,30,(200,200)))
        #VidOutList=cv2.VideoWriter(VidOutFile,-1,30,(np.int(tileList[i][0])+1,1+np.int(tileList[i][1])))
        #VidOutList=cv2.VideoWriter(VidOutFile,fourcc,30,(2048,2048))
    #avi_path = 'D:/data/b/2FishSeries_2/20151125_isolatedVsGroup/expStream2015-11-25T16_45_05_isolatedVsGroup.avi'
    #cap = cv2.VideoCapture('C:/Users/jlarsch/Desktop/testVideo/x264Test.avi')
    cap = cv2.VideoCapture(avi_path)
    img1=cap.read()
    gray = cv2.cvtColor(img1[1], cv2.COLOR_BGR2GRAY)
    allMed=gray.copy()
    fr=0
    
    for i in range(10,100,10):
        cap.set(cv2.CAP_PROP_POS_FRAMES,i)
        image=cap.read()
        gray = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)
        
        allMed=np.dstack((allMed,gray))
        fr+=1
        
    vidMed=np.median(allMed,axis=2)
    height,width,layers=img1[1].shape

    
    fr=0
    
    
    cap.set(cv2.CAP_PROP_POS_FRAMES,6)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    
    while(np.less(fr,90)):
        image=cap.read()
        gray = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)
    
        bgDiv=gray/vidMed
        cv2.imshow('Image',bgDiv)
        k = cv2.waitKey(1) & 0xff
        
        for i in range(np.shape(tileList)[0]):
        #for i in range(1):
        #i=1
            roi=(np.array([tileList[i][2],
            tileList[i][2]+tileList[i][0],
            tileList[i][3],
            tileList[i][3]+tileList[i][1]])).astype('int')
            
            print roi
            
            #xRange=(np.arange(tileList[i][2],tileList[i][2]+tileList[i][0])).astype('int')
            #xRange=np.arange(200)
            #yRange=np.arange(200)
            #yRange=(np.arange(tileList[i][3],tileList[i][3]+tileList[i][1])).astype('int')
            
            #print i,fr
            #print 'ranges:',str(np.shape(xRange)),',',str(np.shape(yRange))
            #print VidOutList[i].get(3)
            #print 'hallo', np.int(tileList[i][0]) +1,1+ np.int(tileList[i][1])
            
            try:
                VidOutList[i].write(bgDiv[roi[0]:roi[1],roi[2]:roi[3]])
                #VidOutList[i].write(bgDiv)
            except:
                print 'error'
                for i in range(np.shape(tileList)[0]):
                #for i in range(1):
                    VidOutList[i].release()   
                
                #raise                
                break
                    
                
        fr += 1
        if k == 27:
            break
     
    cap.release()
    cv2.destroyAllWindows()
    
    for i in range(np.shape(tileList)[0]):
    #for i in range(1):
        #i=1
        VidOutList[i].release()   
    print i

