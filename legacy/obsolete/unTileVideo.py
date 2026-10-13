import matplotlib.pyplot as plt
import numpy as np
import cv2
import tkFileDialog
import wx
import os
import joFishHelper
import time
import pickle
import sys
import subprocess as sp


avi_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b/2FishSeries_2/20151118_gr_vs_het/1'))
#avi_path = 'C:/Users/jlarsch/Desktop/testVideo/x264Test.avi'
#avi_path = 'D:/data/b/2FishSeries_2/20151125_isolatedVsGroup/expStream2015-11-25T16_45_05_isolatedVsGroup.avi'

class UnTileArenaVideo(object):
    #split video recordings of multiple arenas into separate files
    def __init__(self,avi_path):
        
        self.avi_path=avi_path
        head, tail = os.path.split(self.avi_path)
        vp=joFishHelper.getVideoProperties(avi_path)
        self.ffmpeginfo = vp
        self.videoDims = [vp['width'] , vp['height']]
        self.numFrames=int(vp['nb_frames'])
        self.fps=vp['fps']
        # load one frame to select arenas, convert to grayscale
        capture = cv2.VideoCapture(avi_path)
        capture.set(cv2.CAP_PROP_POS_FRAMES,6)
        image = capture.read()
        gray = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)
        
        #wx dialog box
        def ask(parent=None, message='', default_value=''):
            dlg = wx.TextEntryDialog(parent, message, defaultValue=default_value)
            dlg.ShowModal()
            result = dlg.GetValue()
            dlg.Destroy()
            return result

        self.frame = gray
        self.fig = plt.figure(figsize=(6,9))
        self.ax = self.fig.add_subplot(111)
        xaxis = self.frame.shape[1]
        yaxis = self.frame.shape[0]
        
        if ~np.equal(~os.path.isfile(head+'/roi.p'),-2):
            self.im = self.ax.imshow(self.frame[::-1,:], cmap='jet', extent=(0,xaxis,0,yaxis), picker=5)
            self.fig.canvas.mpl_connect('button_press_event', self.onpick1)
            self.fig.canvas.draw()
    
            self.ClickCount=0
            # Initialize wx App
            app = wx.App()
            app.MainLoop()
    
            # Call Dialog
            self.numPicks = np.int(ask(message = 'How Many Arenas?'))
            self.PicksDone=0
            self.x = np.zeros([self.numPicks,4])
            self.y = np.zeros([self.numPicks,4])
            self.roiAll=[] #circular roi for each dish arena
            self.roiSq=[] #rectangular roi around each dish
        else:
            self.roiSq = pickle.load( open(head+"/roi.p", "rb" ) )

    def onpick1(self,event):
        if np.less(self.PicksDone,self.numPicks):
            self.x[self.PicksDone,self.ClickCount]=event.xdata
            self.y[self.PicksDone,self.ClickCount]=event.ydata
            
            if np.less(self.ClickCount,3):
                self.ClickCount += 1
            else: #4 points defined, calculate circle
                self.ClickCount=0
                A=np.array([self.x[self.PicksDone,:],self.y[self.PicksDone,:],np.ones(4)])
                A=A.transpose()
                b=-self.x[self.PicksDone,:]**2-self.y[self.PicksDone,:]**2
                coefs=np.linalg.lstsq(A,b)
                roi=np.zeros(3)
                roi[0]=-coefs[0][0]/2 #center x
                roi[1]=-coefs[0][1]/2 #center y
                roi[2]=np.sqrt(roi[0]**2+roi[1]**2-coefs[0][2]) #radius
                circle=plt.Circle((roi[0],roi[1]),roi[2],color='b',fill=False)
                event.canvas.figure.gca().add_artist(circle)
                event.canvas.draw()
                self.PicksDone +=1
                #print self.PicksDone
                self.roiAll.append(roi)
                wh=roi[2]*2+roi[2]*.1 #width and height of roi around circular roi
                wh=wh+16-np.mod(wh,16) #expand to multiple of 16 for videoCodecs
                self.roiSq.append([wh,wh,roi[0]-wh/2,roi[1]-wh/2])
                
        else:
            print 'all arenas defined'
            plt.close()
            self.roiSq=np.array(list(self.roiSq)).astype('int')
            head, tail = os.path.split(self.avi_path)
            pickle.dump( self.roiSq, open( head+"/roi.p", "wb" ) )
            
    def videoSplit(self):
        numAr=np.shape(self.roiSq)[0]
        head, tail = os.path.split(self.avi_path)
        VidOutList=[]
        FFMPEG_PATH = 'ffmpeg.exe'
        
        for i in range(numAr):
            #create subdirectories for split output
            directory=head+'/'+ str(i+1)+'/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            VidOutFile=(directory+'split_'+str(i+1)+'_'+tail)
            fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
            VidOutList.append(cv2.VideoWriter(VidOutFile,fourcc,30,(np.int(self.roiSq[i][0]),np.int(self.roiSq[i][1]))))
        
        cap = cv2.VideoCapture(self.avi_path)
        img1=cap.read()
        gray = cv2.cvtColor(img1[1], cv2.COLOR_BGR2GRAY)
        allMed=gray.copy()
        
        for i in range(10,self.numFrames-2,np.round(self.numFrames/9)): #use 9 images to calculate median
            cap.set(cv2.CAP_PROP_POS_FRAMES,i)
            print i
            image=cap.read()
            gray = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)
            
            allMed=np.dstack((allMed,gray))
            
        vidMed=np.median(allMed,axis=2)
        print 'background collected, beginning UnTiling...'
        height,width,layers=img1[1].shape     
        fr=0
        cap.set(cv2.CAP_PROP_POS_FRAMES,6)
        #cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        
        #while(np.less(fr,self.numFrames-2)):
        t1=time.time()
        bgDiv=gray/vidMed
        sys.stdout.write( bgDiv.tostring() )
        cmd='ffmpeg.exe -f rawvideo -pix_fmt gray -s 2048x2048 -r 30 -i - -an -f avi -r 30 foo.avi'
        sp.call(cmd,shell=True)
        
        while(np.less(fr,50)):
            image=cap.read()
            print fr
            gray = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)
        
            bgDiv=gray/vidMed
            #v2.imshow('Image',bgDiv)
           
            sys.stdout.write( bgDiv.tostring() )
            #k = cv2.waitKey(1) & 0xff
            
            for i in range(numAr):
                roi=(np.array([self.roiSq[i][2],
                self.roiSq[i][2]+self.roiSq[i][0],
                self.roiSq[i][3],
                self.roiSq[i][3]+self.roiSq[i][1]])).astype('int')
                
                #try:
                    #VidOutList[i].write(bgDiv[roi[0]:roi[1],roi[2]:roi[3]])
                #except:
                #    print 'error'
                #    for i in range(numAr):
                #        VidOutList[i].release()             
                #    break
            fr += 1
            #if k == 27:
            #    break
        
        t2=time.time()
        print 'this took this much time:', t2-t1
        cap.release()
        cv2.destroyAllWindows()
        
        for i in range(numAr):
            VidOutList[i].release()   
        
Scl=UnTileArenaVideo(avi_path)
