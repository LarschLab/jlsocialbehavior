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
import os
import matplotlib.pyplot as plt
import models.experiment as xp
import glob
import numpy as np
import pandas as pd
import datetime
import functions.CameraInterceptCorrection as cic

class settings(object):

    def __init__(self, startFrame=0,
                 endFrame=108000+6000):
        self.startFrame=startFrame
        self.endFrame=endFrame      
        self.currFrame=108000-3000
        self.run=False
        self.vidRec=False
        self.haveVid=False
        self.drawStim=True
        self.drawAn=False
        self.drawStimPath=False
        self.drawAnPath=False
        self.drawAnSize=8
        self.drawStimSize=4
        self.drawAnPathSize=2
        self.drawStimPathSize=2
        self.drawArenas=True
        self.drawTime=True
        self.drawAnCol=(0,0,0)
        self.drawStimCol=(0,0,1)
        self.drawTailLen=10
        self.drawTailStep=10.0
        self.drawVideoFrame=True
        self.PLcode=True

class vidGui(object):

    window_name = "vidGUI"
    def __init__(self, path,e1,df,ROIdf,PLdf):
        self.settings=settings()
        self.df=df
        self.PLdf=PLdf
        self.ROIdf=ROIdf
        self.skipNan=121#e1.skipNanInd
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.im=[]
        self.rawTra=e1.rawTra.values.copy()
        cv2.namedWindow(self.window_name,cv2.WINDOW_NORMAL)

        cv2.setMouseCallback(self.window_name, self.startStop)
        cv2.createTrackbar("startFrame", self.window_name,
                           self.settings.startFrame, 108000-6000,
                           self.set_startFrame)
        cv2.createTrackbar("endFrame", self.window_name,
                           self.settings.endFrame, 108000+6000,
                           self.set_endFrame)
        cv2.createTrackbar("currFrame", self.window_name,
                           self.settings.currFrame, 108000+3000,
                           self.set_currFrame)
    
    def startStop(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.settings.run=True
            while self.settings.run:
                f=self.settings.currFrame
                f+=1
                key = cv2.waitKey(500) & 0xFF
            
                if key == ord("x"):
                    if self.settings.haveVid:
                        self.vOut.release()
                    cv2.destroyAllWindows()
                 
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
        
         

        
        xMax=2048.0 
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.settings.currFrame)
        self.im=255-self.cap.read()[1]
        if self.settings.drawVideoFrame:
            mi=150
            ma=250
            self.im=((self.im - mi)/(ma - mi))*255
            self.im[self.im>255]=255
            self.im[self.im<0]=0
        else:
            self.im[:]=255
        self.im=self.im.astype('uint8')
        fs= self.settings.currFrame-np.mod(self.settings.currFrame,5)
        self.currEpi=self.df['episode'].ix[np.where(self.df['epStart']>fs)[0][0]-1]
        r=np.arange(fs-self.settings.drawTailLen,fs,self.settings.drawTailStep).astype('int')
        
        #DRAW path history
        #loop over all arenas
        
        for ar in np.arange(ROIdf.shape[0]):
            
            offset=self.ROIdf.values[ar]
            #print(ar,offset)
            pAn=self.rawTra[:,ar*3:ar*3+2].copy()
            xoff=offset[0]
            yoff=offset[1]
            xx,yy=cic.deCorrectFish(pAn[:,0],pAn[:,1],xoff,yoff,xMax,53.)
            pAn[:,0]=xx+xoff
            pAn[:,1]=yy+yoff
            
            if self.settings.PLcode:
                pairListNr = int(self.currEpi[:2])
                pairList = PLdf.values[pairListNr * 16:(pairListNr + 1) * 16, ar]
            stimAn=np.where(pairList)[0][0]
            pSt=self.rawTra[:,stimAn*3:stimAn*3+2]+offset[:2]
            
            # draw path history 
            for f in r:
                opacity=250*((fs-f)/float(self.settings.drawTailLen))
                if self.settings.drawStimPath:
                    center=tuple(pSt[f,:].astype('int'))
                    #print('before')
                    c=tuple(np.array([opacity if x==0 else 250. for x in self.settings.drawStimCol]))
                    #print(center,  self.settings.drawStimPathSize, c,self.settings.drawStimCol)
                    cv2.circle(self.im, center,  self.settings.drawStimPathSize, c, -1) 
                    #print('drawn')
                if self.settings.drawAnPath:
                    center=tuple(pAn[f,:].astype('int'))
                    c=tuple(np.array([opacity if x==0 else 250. for x in self.settings.drawAnCol]))
                    cv2.circle(self.im, center,  self.settings.drawAnPathSize, c, -1) 
                    #print('drawn2')
    #            
    #        #DRAW Current stimulus position
            if self.settings.drawStim:
                
                center=tuple(pSt[f,:].astype('int'))
                cv2.circle(self.im, center, self.settings.drawStimSize, self.settings.drawStimCol, -1)
            if self.settings.drawAn:
                center=tuple(pAn[f,:].astype('int'))
                cv2.circle(self.im, center, self.settings.drawAnSize, self.settings.drawAnCol, -1)            

    #        #DRAW DISH BOUNDARIES
            if self.settings.drawArenas:
                center=tuple(offset[3:5])
                cv2.circle(self.im, center, offset[-1], (0,0,0), 2)

            if self.settings.drawTime:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(self.im,self.currEpi,(380,20), font, 0.6,(0,0,0),2)  
                
                frInfo='Frame #: '+str(self.settings.currFrame)
                cv2.putText(self.im,frInfo,(380,40), font, 0.4,(0,0,0),2)
                
                miliseconds=self.settings.currFrame*(100/3.0)
                seconds,ms=divmod(miliseconds, 1000)
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                timeInfo= "%02d:%02d:%02d:%03d" % (h, m, s,ms)
                cv2.putText(self.im,timeInfo,(380,60), font, 0.4,(0,0,0),2)
                timeInfo2='(hh:mm:ss:ms)'
                cv2.putText(self.im,timeInfo2,(380,80), font, 0.4,(0,0,0),2)
    

        
        cv2.imshow(self.window_name, self.im)
        if self.settings.vidRec:
           
            wr=self.im.astype('uint8')

            self.vOut.write(wr)

        
#p='D:\\data\\b\\2017\\20170131_VR_skypeVsTrefoil\\01_skypeVsTrefoil_blackDisk002\\'
#p='C:\\Users\\johannes\\Dropbox\\20170131_VR_skypeVsTrefoil_01\\'
#p='C:\\Users\\johannes\\Dropbox\\20170710124615\\'; avi_path=p+'out_id0_30fps_20170710124615.avi' #frame: 82800
#p='D:\\data\\b\\2017\\20170921_SkypePairPermutations\\'; avi_path=p+'out_id0_30fps_20170921120214.avi'#frame 53700

p='E:\\b\\medaka\\20200110_15animals_skypeAndKnot\\'; avi_path=p+'out_id0_30fps_20200110161244.avi'#frame 53700
#avi_path = filedialog.askopenfilename(initialdir=os.path.normpath(p))   



rereadTxt=1


p, tail = os.path.split(avi_path)
tp=glob.glob(p+'\\Position*.txt')
    
if rereadTxt:
    e1=xp.experiment(tp[0])
    
csvFileOut=glob.glob(tp[0][:-4]+'_siSummary_epi*.csv')[0]
ROIfn=glob.glob(p+'\\ROI*')[0]
PLfn=glob.glob(p+'\\PL_*')[0]
df=pd.read_csv(csvFileOut,index_col=0,sep=',')[['epStart','episode']]
ROIdf=pd.read_csv(ROIfn,index_col=None,sep=' ',header=None)
PLdf=pd.read_csv(PLfn,index_col=None,sep=' ',header=None)
a=vidGui(avi_path,e1,df,ROIdf,PLdf)
