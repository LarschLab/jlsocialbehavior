# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 16:44:53 2018

@author: jlarsch
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 10:55:23 2017

@author: jlarsch
"""

import cv2
#from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import models.experiment as xp
import glob
import numpy as np
import pandas as pd
import datetime

class settings(object):
    """
    A stereo pair of calibrated cameras.
 
    Should be initialized with a context manager to ensure that the camera
    connections are closed properly.
    """
    def __init__(self, startFrame=0,
                 endFrame=10):
        self.startFrame=startFrame
        self.endFrame=endFrame      
        self.currFrame=5000
        self.run=False
        self.vidRec=False
        self.haveVid=False

class vidGui(object):
    """
    A class for tuning Stereo BM settings.
 
    Display a normalized disparity picture from two pictures captured with a
    ``CalibratedPair`` and allow the user to manually tune the settings for the
    stereo block matcher.
    """
    #: Window to show results in
    window_name = "vidGUI"
    def __init__(self, path,e1,e2,df):
        """Initialize tuner with a ``CalibratedPair`` and tune given pair."""
        #: Calibrated stereo pair to find Stereo BM settings for
        self.settings=settings()
        self.df=df
        self.skipNan=121#e1.skipNanInd
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.im=[]
        self.t1=e1.rawTra.values[:,0:2].copy()
        self.t2=e2.rawTra.values[:,0:2].copy()
        self.t1b=e1.rawTra.values[:,2:3].copy()
        self.t2b=e2.rawTra.values[:,2:3].copy()
        self.t1[:,0]=self.t1[:,0]+512        
        self.t1b[:,0]=self.t1b[:,0]+512  
        #vcv2.namedWindow(self.window_name)
        cv2.namedWindow(self.window_name,cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.startStop)
        cv2.createTrackbar("startFrame", self.window_name,
                           self.settings.startFrame, 1000,
                           self.set_startFrame)
        cv2.createTrackbar("endFrame", self.window_name,
                           self.settings.endFrame, 50000,
                           self.set_endFrame)
        cv2.createTrackbar("currFrame", self.window_name,
                           self.settings.currFrame, 10000,
                           self.set_currFrame)
    
    def startStop(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.settings.run=True
            while self.settings.run:
                f=self.settings.currFrame
                f+=1
                
            
                key = cv2.waitKey(50) & 0xFF
                 
                if key == ord("v"):
                  
                    self.settings.vidRec = not self.settings.vidRec
                    
                    if self.settings.vidRec:
                        if not self.settings.haveVid:
                                             
                            p, tail = os.path.split(self.path)
                            fn=p+"\\episodeVid_frame_%07d_%s.avi"%(f,self.currEpi)
                            fn=os.path.normpath(fn)
                            fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
                            fps=30
                            self.vOut = cv2.VideoWriter(fn,fourcc,fps,self.im.transpose(1,0,2).shape[0:2],True)
                            self.settings.haveVid=True
                    else:
                        if self.settings.haveVid:
                            self.vOut.release()
                            self.settings.haveVid=False
     
                if key == ord("s"):
                    self.settings.run=False
                cv2.setTrackbarPos("currFrame", self.window_name,f)
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.settings.run=False
                      
    def set_startFrame(self, f1):
        self.settings.startFrame = f1
        self.updateFrame()

    def set_endFrame(self, f2):
        self.settings.endFrame = f2
        self.updateFrame()

    def set_currFrame(self, f3):
        
        self.settings.currFrame = f3
        self.updateFrame()
    def updateFrame(self):
        
         
        tail=500
        tailStep=5.0
        pathDotSize=1
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.settings.currFrame)
        self.im=255-self.cap.read()[1]
        fs= self.settings.currFrame-np.mod(self.settings.currFrame,5)
        self.currEpi=self.df['episode'].ix[np.where(self.df['epStart']>fs)[0][0]-1][1:]
        r=np.arange(fs-tail,fs,tailStep).astype('int')
        
        #DRAW path history
        for f in r:
            opacity=((fs-f)/float(tail))
            
            center=tuple(self.t1b[f,:].astype('int'))
            cv2.circle(self.im, center, pathDotSize, (opacity*255,opacity*255,255), -1) 
            center=tuple(self.t1[f,:].astype('int'))
            cv2.circle(self.im, center, pathDotSize, (opacity*255,opacity*255,opacity*255), -1) 
             
            center=tuple(self.t2[f,:].astype('int'))
            cv2.circle(self.im, center, pathDotSize, (opacity*255,opacity*255,opacity*255), -1) 
            center=tuple(self.t2b[f,:].astype('int'))
            cv2.circle(self.im, center, pathDotSize, (255,opacity*255,opacity*255), -1)
            
            
        #DRAW Current animal positions    
        center=tuple(self.t1[f,:].astype('int'))
        cv2.circle(self.im, center, 4, (0,0,0), -1)
        if 'skype' in self.currEpi:
            cv2.circle(self.im, center, 6, (255,opacity*255,opacity*255), 1)
        center=tuple(self.t2[f,:].astype('int'))
        cv2.circle(self.im, center, 4, (0,0,0), -1)
        if 'skype' in self.currEpi:
            cv2.circle(self.im, center, 6, (opacity*255,opacity*255,255), 1)
        
        
        #DRAW DISH BOUNDARIES
        center=tuple((256,256))
        cv2.circle(self.im, center, 240, (0,0,0), 2)
        center=tuple((256+512,256))
        cv2.circle(self.im, center, 240, (0,0,0), 2)
        center=tuple((256+2*512,256))
        cv2.circle(self.im, center, 240, (0,0,0), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.im,self.currEpi,(450,20), font, 0.6,(0,0,0),2)  
        
        frInfo='Frame #: '+str(self.settings.currFrame)
        cv2.putText(self.im,frInfo,(450,40), font, 0.4,(0,0,0),2)
        
        miliseconds=self.settings.currFrame*(100/3.0)
        seconds,ms=divmod(miliseconds, 1000)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        timeInfo= "%02d:%02d:%02d:%03d" % (h, m, s,ms)
        cv2.putText(self.im,timeInfo,(450,60), font, 0.4,(0,0,0),2)
        timeInfo2='(hh:mm:ss:ms)'
        cv2.putText(self.im,timeInfo2,(450,80), font, 0.4,(0,0,0),2)

        cv2.line(self.im, tuple((1024,0)), tuple((1024,512)), (0,0,0))        
        
        cv2.putText(self.im,"Arena 1",(220,60), font, 0.6,(0,0,0),2)
        cv2.putText(self.im,"Arena 2",(220+512,60), font, 0.6,(0,0,0),2)
        cv2.putText(self.im,"overlay 1 & 2",(200+1024,60), font, 0.6,(0,0,0),2)
        
        cv2.imshow(self.window_name, self.im)
        if self.settings.vidRec:
           
            wr=self.im.astype('uint8')

            self.vOut.write(wr)

        
p='D:\\data\\b\\2017\\20170131_VR_skypeVsTrefoil\\01_skypeVsTrefoil_blackDisk002\\'
 

rereadTxt=1
if rereadTxt:
    #avi_path = filedialog.askopenfilename(initialdir=os.path.normpath(p))   
    avi_path=p+'out_id0_30fps_20170131090314.avi'
    p, tail = os.path.split(avi_path)
    tp=glob.glob(p+'\\Position*.txt')
    e1=xp.experiment(tp[0])
    e2=xp.experiment(tp[1])    
    
csvFileOut=tp[0][:-4]+'_siSummary_epi'+str(10.0)+'.csv'
df=pd.read_csv(csvFileOut,index_col=0,sep=',')[['epStart','episode']]
    
a=vidGui(avi_path,e1,e2,df)
