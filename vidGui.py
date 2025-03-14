# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 10:55:23 2017

@author: jlarsch


This script creates demo videos for loom experiments. Works with Miguel's most recent competition
experiments as of May 2019.

Requires the raw video file, it's posText output file and the ROIdef file all in the same folder.

Run this script in Spyder (for Python 3)

In the file dialog, select the video file.

Now use the slider 'currFrame' so slide to the frame where you want to demo video to start.

Use left mouse click into the video to begin video playback, right mouse click to stop.

WHILE PLAYBACK IS RUNNING, hit keyboard key 'v' (for video). This will begin saving the current playback.
(You won't notice any change, except maybe playback slows down).

When the video has reached the desired frame, hit keyboard key 'v' again to end the recording.

You can now stop the playback using the mouse right click.

You should now have a new file in the video folder with the starting frame in the file name.

You can now slide to a new frame and record another video without re-running the script.

When the order of clicking playback start and recording the video is messed up,
python can get stuck with an empty video file that it struggles to release.
In this case, it is easiest to restart the kernel and re-run the script.

You can also define the starting frame precisely when running the script via  self.currFrame='frame number'.

Once the video gui is loaded, you can toggle rereadTxt=0 and rere-run the script without closing the gui.
This way, you can set currFrame directly to a desired value.

length = 50000 (in class vidGui(object): #This limits how far you can scroll into the video.
In the past, the gui struggled to seek deeply into videos but 50.000 or even 100.000 frames should be OK.


"""



import cv2
from tkinter import filedialog
import os
import glob
import numpy as np
import pandas as pd

rereadTxt=1 #set to 1 for initial start, can toggle to 0 once data is loaded.

class settings(object):

    def __init__(self, startFrame=0,
                 endFrame=10):
        self.startFrame=startFrame
        self.endFrame=endFrame      
        self.currFrame=20000 # Can define the starting frame for the demo recording here.
        self.run=False
        self.vidRec=False
        self.haveVid=False

class vidGui(object):

    window_name = "vidGUI"
    def __init__(self, path,anMat,sMat=[],df_roi=[]):

        self.settings=settings()
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.im=[]
        self.df_roi=df_roi
        self.anMat=anMat
        self.sMat=sMat
        cv2.namedWindow(self.window_name,cv2.WINDOW_AUTOSIZE)
        self.desiredWidth=1024
        self.desiredheight=1024
        

        length = 50000 #This limits how far you can scroll into the video. see notes above.
        cv2.setMouseCallback(self.window_name, self.startStop)

        cv2.createTrackbar("currFrame", self.window_name,
                           self.settings.currFrame, length,
                           self.set_currFrame)
    def startStop(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.settings.run=True
            while self.settings.run:
                f=self.settings.currFrame
                f+=1
                
            
                key = cv2.waitKey(1) & 0xFF
                 
                if key == ord("v"): #for video recording: toggle start-stop
                  
                    self.settings.vidRec = not self.settings.vidRec
                    
                    if self.settings.vidRec:
                        if not self.settings.haveVid:
                                             
                            p, tail = os.path.split(self.path)
                            fn=p+"\\episodeVid_frame_%07d.avi"%(f)
                            fn=os.path.normpath(fn)
                            fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
                            fps=30
                            self.vOut = cv2.VideoWriter(fn,fourcc,fps,self.im.transpose(1,0,2).shape[0:2],True)
                            self.settings.haveVid=True
                    else:
                        if self.settings.haveVid:
                            self.vOut.release()
                            self.settings.haveVid=False
     
                if key == ord("s"): #start-stop video: redundant with mouse click.
                    self.settings.run=False
                cv2.setTrackbarPos("currFrame", self.window_name,f)
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.settings.run=False
                      

    def set_currFrame(self, f3):
        
        self.settings.currFrame = f3
        self.updateFrame()
    def updateFrame(self):
        
        stimDotSize=4
        stimDotSize2=4

        # some of the commented stuff below can be used to also draw the path of the animal.         
        #tail=1 #length of animal path
        #tailStep=1.0# interval between path data points to plot
        #dotScale=1
        #pathDotSize=0#4 
        #animalDotSize=0#6 #not showing animal path for now
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.settings.currFrame)
        self.im=255-self.cap.read()[1]
        #fs= self.settings.currFrame -np.mod(self.settings.currFrame,tailStep)
        #self.currEpi=self.df['episode'].ix[np.where(self.df['epStart']>fs)[0][0]-1][1:]
        #r=np.arange(fs-tail,fs,tailStep).astype('int')
        s=sMat[0,self.settings.currFrame,0]
        
        if ~np.isnan(s):
            stimDotSize=int(s)
            stimDotSize2=int(sMat[0,self.settings.currFrame,1])
        #DRAW path history
        #for f in r:
        #for f in [self.settings.currFrame]:
        f=self.settings.currFrame
            #opacity=((fs-f)/float(tail))
            
            #for an in range(9):
            #    center=tuple(self.anMat[an,f,[0,1]].astype('int'))
                #cv2.circle(self.im, center,pathDotSize , (opacity*255,opacity*255,255), -1) 
                #center=tuple(self.anMat[an,f,[2,3]].astype('int'))
                #cv2.circle(self.im, center, stimDotSize, (opacity*255,opacity*255,opacity*255), -1) 
                #center=tuple(self.anMat[an,f,[4,5]].astype('int'))
                #cv2.circle(self.im, center, stimDotSize2, (opacity*255,opacity*255,opacity*255), -1)              

        
        #DRAW Current frame animal positions 
        for an in range(9):
            opacity=0
            #center=tuple(self.anMat[an,f,[0,1]].astype('int'))
            #cv2.circle(self.im, center, animalDotSize, (0,1,0), -1)
            center=tuple(self.anMat[an,f,[2,3]].astype('int'))
            cv2.circle(self.im, center, stimDotSize, (opacity*255,opacity*255,opacity*255), -1) 
            center=tuple(self.anMat[an,f,[4,5]].astype('int'))
            cv2.circle(self.im, center, stimDotSize2, (opacity*255,opacity*255,opacity*255), -1)  

        #if 'skype' in self.currEpi:
        #    cv2.circle(self.im, center, 6, (255,opacity*255,opacity*255), 1)
        
        #DRAW DISH BOUNDARIES
        for index, row in self.df_roi.iterrows():
            center=tuple((row.x_center,row.y_center))
            rad=row.radius
            cv2.circle(self.im, center, rad, (0,0,0), 2)
        
        #Add frame and size info Text to video frame.
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(self.im,self.currEpi,(450,20), font, 0.6,(0,0,0),2)  
        
        frInfo='Frame #: '+str(self.settings.currFrame)
        cv2.putText(self.im,frInfo,(500,40), font, 0.4,(0,0,0),2)
        
        miliseconds=self.settings.currFrame*(100/3.0)
        seconds,ms=divmod(miliseconds, 1000)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        timeInfo= "%02d:%02d:%02d:%03d" % (h, m, s,ms)
        cv2.putText(self.im,timeInfo,(500,60), font, 0.4,(0,0,0),2)
        timeInfo2='(hh:mm:ss:ms)'
        cv2.putText(self.im,timeInfo2,(500,80), font, 0.4,(0,0,0),2)

        #cv2.line(self.im, tuple((1024,0)), tuple((1024,512)), (0,0,0))        
        #cv2.putText(self.im,"Arena 1",(220,60), font, 0.6,(0,0,0),2)
        #cv2.putText(self.im,"Arena 2",(220+512,60), font, 0.6,(0,0,0),2)
        cv2.putText(self.im,'DotSize1: ~'+str(stimDotSize/2)+' mm',(650,60), font, 0.6,(0,0,0),2)
        cv2.putText(self.im,'DotSize2: ~'+str(stimDotSize2/2)+' mm',(650,80), font, 0.6,(0,0,0),2)
        #cv2.resizeWindow(self.window_name, self.desiredWidth,self.desiredheight)
        #newWidth=1024
        #r = newWidth / self.im.shape[1]
        #dim = (newWidth, int(self.im.shape[0] * r))
 
        # perform the actual resizing of the image and show it
        dim=(1024,1024) #for screen scaling. this does not affect the saved video
        resized = cv2.resize(self.im, dim)
         
        cv2.imshow(self.window_name, resized)
                
        if self.settings.vidRec:
           
            wr=self.im.astype('uint8')
            self.vOut.write(wr)

if rereadTxt:
    lines=67                        #for compatibility with older files
    empty=np.repeat('NaN',lines)    #for compatibility with older files
    empty=' '.join(empty)           #for compatibility with older files
    p='R:\\johannes\\from Miguel\\video to annotate\\' #starting directory for file dialog
    avi_path = filedialog.askopenfilename(initialdir=os.path.normpath(p))    
    p, tail = os.path.split(avi_path)
    txt_path=glob.glob(p+'\\Position*.txt')[0]
    roi_path=glob.glob(p+'\\ROI*.csv')[0]
    
    df_roi=pd.read_csv(roi_path,
        names=['x_topLeft',
        'y_topLeft',
        'widthHeight',
        'x_center',
        'y_center',
        'radius'],
        delim_whitespace=True)
    
    with open(txt_path) as f:
        mat=np.loadtxt((x if len(x)>6 else empty for x in f ),delimiter=' ')
    #epiLen=int(np.median(np.diff(np.where(mat[:-1,-1]!=mat[1:,-1]))))
    epiLen=int(np.median(np.diff(np.where((mat[:-1,-1]!=mat[1:,-1]) * ((mat[:-1,-1])==0)))))
    print(epiLen)
    #ind=np.arange(149+epiLen,mat.shape[0],epiLen)
    #fixCol=[3,4,5,6,10,11,12,13,17,18,19,20,24,25,26,27,28,29]
    #for fc in fixCol:
    #    mat[ind,fc]=mat[ind+1,fc]
    
    anMat=[]
    sMat=[]
    for an in range(9):
        #pull columns corresponding to position of animal and its two stimuli
        tmp=np.array(mat[:,an*7+np.array([0,1,3,4,5,6])]) 
        #Add arena offset
        tmp[:,[0,2,4]]=tmp[:,[0,2,4]]+df_roi.x_topLeft[an]
        tmp[:,[1,3,5]]=tmp[:,[1,3,5]]+df_roi.y_topLeft[an]
        anMat.append(tmp)
        #pull size for both dots - this is the same for animals.
        tmp=np.array(mat[:,[-6,-5]]) #Size columns for the two stimuli
        sMat.append(tmp)
        #df['episode']=np.repeat(np.arange(mat.shape[0]/float(epiLen)),epiLen)
        #dfAll.append(df.copy())
    anMat=np.array(anMat)
    sMat=np.array(sMat)
    #e1=xp.experiment(avi_path,tp[0])
    #e2=xp.experiment(avi_path,tp[1])    
    
#csvFileOut=tp[0][:-4]+'_siSummary_epi'+str(10)+'.csv'
#df=pd.read_csv(csvFileOut,index_col=0,sep=',')[['epStart','episode']]
    
a=vidGui(avi_path,anMat,sMat,df_roi)