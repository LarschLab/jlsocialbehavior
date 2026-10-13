import numpy as np
import subprocess
import os
import matrixUtilities_joh as mu
import random
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as sta
import plotFunctions_joh as johPlt
import cv2
from matplotlib.backends.backend_pdf import PdfPages

class ExperimentMeta(object):
    #This class collects paths, arena and video parameters
    def __init__(self,path,arenaDiameter_mm=100):
        self.arenaDiameter_mm = arenaDiameter_mm
        self.arenaCenterPx = [0,0] #assign later from class 'Pair'
        self.pxPmm = 0 #assign later from class 'Pair'
        
        #If a video file name is passed, collect video parameters
        if path.endswith('.avi') or path.endswith('.mp4'):
            self.aviPath = path
            #get video meta data
            vp=getVideoProperties(path) #video properties
            self.ffmpeginfo = vp
            self.videoDims = [vp['width'] , vp['height']]
            self.numFrames=vp['nb_frames']
            self.fps=vp['fps']
            #self.date=vp['TAG:date']
        else:
            self.fps=30
            
        
        #concatenate dependent file paths (trajectories, pre-analysis)
        head, tail = os.path.split(path)
        head=os.path.normpath(head)
        self.trajectoryPath = os.path.join(head,'trajectories_nogaps.mat')
        self.dataPath = os.path.join(head,'analysisData.mat')
        
def getVideoProperties(aviPath):
    #read video metadata via ffprobe and parse output
    #can't use openCV because it reports tbr instead of fps (frames per second)
    cmnd = ['c:/ffmpeg/bin/ffprobe', '-show_format', '-show_streams', '-pretty', '-loglevel', 'quiet', aviPath]
    p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err =  p.communicate()
    decoder_configuration = {}
    for line in out.splitlines():
        if '=' in line:
            key, value = line.split('=')
            decoder_configuration[key] = value
    
    #frame rate needs special treatment. calculate from parsed str argument        
    nominator,denominator=decoder_configuration['avg_frame_rate'].split('/')
    decoder_configuration['fps']=int(float(nominator) / float(denominator))
    return decoder_configuration
    

def getMedVideo(aviPath,FramesToAvg,saveFile):
    cap = cv2.VideoCapture(aviPath)
    head, tail = os.path.split(aviPath)
    vp=getVideoProperties(aviPath)
    videoDims = tuple([int(vp['width']) , int(vp['height'])])
    print videoDims
    #numFrames=int(vp['nb_frames'])
    numFrames=50000
    img1=cap.read()
    img1=cap.read()
    img1=cap.read()
    img1=cap.read()
    gray = cv2.cvtColor(img1[1], cv2.COLOR_BGR2GRAY)
    allMed=gray.copy()
    for i in range(10,numFrames-2,np.round(numFrames/FramesToAvg)): #use FramesToAvg images to calculate median
        cap.set(cv2.CAP_PROP_POS_FRAMES,i)
        image=cap.read()
        print i
        gray = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)  
        allMed=np.dstack((allMed,gray))
        
    vidMed=np.median(allMed,axis=2)

    if saveFile:
        ImOutFile=(head+'/bgMed.tif')
        cv2.imwrite(ImOutFile,vidMed)
        return 1
    else:
        return vidMed


class Pair(object):
    #Class for trajectories of one pair of animals.
    #Calculation and storage of dependent time series: speed, heading, etc.
    def __init__(self, trajectories, expInfo):
        self.positionPx=trajectories
        maxPixel=np.nanmax(self.positionPx,0)
        minPixel=np.nanmin(self.positionPx,0)
        arenaDiameterPx=np.mean(maxPixel-minPixel)
        #expInfo.pxPmm=arenaDiameterPx/expInfo.arenaDiameter_mm
        expInfo.pxPmm=8.6
        expInfo.arenaCenterPx=np.mean(maxPixel-(arenaDiameterPx/2),axis=0)
        
        self.position=(self.positionPx-expInfo.arenaCenterPx) / expInfo.pxPmm
        self.d_position=np.diff(self.position,axis=0)
        self.dd_position=np.diff(self.d_position,axis=0)
        self.travel=np.sqrt(self.d_position[:,:,0]**2 + self.d_position[:,:,1]**2)
        self.speed=self.travel/expInfo.fps
        self.totalTravel=np.sum(np.abs(self.travel),axis=0)
        self.accel=np.diff(self.speed,axis=0)
        self.heading=np.transpose(mu.cart2pol(self.d_position[:,:,0],self.d_position[:,:,1]),[1,2,0]) #heading[0] = heading, heading[1] = speed
        #heading: pointing right = 0, pointung up = pi/2      
        self.d_heading=np.diff(self.heading[0],axis=0)
        
        #absolute inter animal distance IAD
        dist=self.position[:,0,:]-self.position[:,1,:]
        self.IAD = np.sqrt(dist[:,0]**2 + dist[:,1]**2)
        self.IAD_m=np.nanmean(self.IAD)
        
        #relative distance between animals
        self.neighborMat,self.relPosPolRotCart,self.relPos= getRelativeNeighborPositions(self.position,self.heading)
        
        #force map between animals
        self.ForceMat = getNeighborForce(self.relPosPolRotCart,self.accel)
        
        #self.dwellH,self.dwellL,self.dwellHTL,self.dwellLTH=getShoalDwellTimes(self.IAD)
        
def getRelativeNeighborPositions(position,heading):
    pos=position[1:,:,:].copy() #obtain a new copy rather than using reference
    
    relPos1=pos[:,0,:]-pos[:,1,:] #position of animal 1 relative to animal 2. (where animal 2 has a neighbor)
    relPos2=pos[:,1,:]-pos[:,0,:]  
    
    relPos=np.transpose([relPos1,relPos2],[1,0,2])
    
    relPosPol=np.transpose(mu.cart2pol(relPos[:,:,0],relPos[:,:,1]),[1,2,0])
    relPosPolRot=relPosPol.copy()
    #rotate 
    relPosPolRot[:,0,0]=relPosPol[:,0,0]-(heading[:,1,0])
    relPosPolRot[:,1,0]=relPosPol[:,1,0]-(heading[:,0,0])
    relPosPolRotCart=mu.pol2cart(relPosPolRot[:,:,0],relPosPolRot[:,:,1])
    relPosPolRotCart=np.transpose(relPosPolRotCart,[1,2,0])
    
    mapBins=np.arange(-31,32)
    neighborMat=np.zeros([62,62,2])
    #creates the neighbormat for animal 2 (where animal1 was)
    #this map seems flipped both horizontally and vertically! vertical is corrected at plotting.
    neighborMat[:,:,1]=np.histogramdd(relPosPolRotCart[:,0,:],bins=[mapBins,mapBins])[0]
    #creates the neighbormat for animal 1 (where animal2 was)
    neighborMat[:,:,0]=np.histogramdd(relPosPolRotCart[:,1,:],bins=[mapBins,mapBins])[0]
    return neighborMat,relPosPolRotCart, relPos
    
def getNeighborForce(position,acceleration):
    #position here should be transformed (rotated, relative). [:,0,:] is the position of 
    mapBins=np.arange(-31,32)
    force_t=np.zeros([62,62,2])
    #animal 2 
    force_t[:,:,1]=sta.binned_statistic_2d(position[1:,0,0],position[1:,0,1],acceleration[:,1],bins=[mapBins,mapBins])[0]
    #animal 1
    force_t[:,:,0]=sta.binned_statistic_2d(position[1:,1,0],position[1:,1,1],acceleration[:,0],bins=[mapBins,mapBins])[0]
    return force_t

def getShoalDwellTimes(IAD):
    IADsm=mu.runningMean(IAD,30)
    lowThreshold=np.nanmax(IADsm)/1.5
    
    lowIAD=(np.less(IAD,lowThreshold)).astype(int)
    hiIAD=(np.greater(IAD,lowThreshold)).astype(int)
    
    #transitions from low to high inter animal distance and vice versa
    LowToHigh=np.where(np.equal((hiIAD[:-1]-hiIAD[1:]),-1))[0]
    HighToLow=np.where(np.equal((lowIAD[:-1]-lowIAD[1:]),-1))[0]
    
    #number of transitions to use below
    maxL=np.min([np.shape(HighToLow)[0],np.shape(LowToHigh)[0]])-2
    
    #How long are High and low dwell times?
    #calculate from transition times. Order depends on starting state of data
    
    if HighToLow[0]>LowToHigh[0]: #meaning starting low
        HiDwell=HighToLow[0:maxL]-LowToHigh[0:maxL]
        LowDwell=LowToHigh[1:maxL]-HighToLow[0:maxL-1]
    else: #meaning starting high
        HiDwell=HighToLow[1:maxL]-LowToHigh[0:maxL-1]
        LowDwell=LowToHigh[0:maxL]-HighToLow[0:maxL]

    return HiDwell,LowDwell,HighToLow,LowToHigh

class shiftedPair(object):
    #Class to calculate null hypothesis time series using shifted pairs of animals
    def __init__(self, tra,expInfo):
        self.nRuns=10
        self.minShift=5*60*expInfo.fps
        self.sPair=[]
        #generate nRuns instances of Pair class with one animal time shifted against the other
        for i in range(self.nRuns):
            traShift=tra.positionPx
            shiftIndex=int(random.uniform(self.minShift,traShift.shape[0]-self.minShift))
            #time-rotate animal 0, keep animal 1 as is
            traShift[:,0,:]=np.roll(traShift[:,0,:],shiftIndex,axis=0)
            self.sPair.append(Pair(traShift,expInfo))
        #calculate mean and std IAD for the goup of shifted instances
        self.spIAD_m = np.nanmean([x.IAD_m for x in self.sPair])
        self.spIAD_std = np.nanstd([x.IAD_m for x in self.sPair])
       

       
       
class experiment(object):
    #Class to collect, store and plot data belonging to one experiment
    def __init__(self,path):
        self.expInfo=ExperimentMeta(path)
        mat=scipy.io.loadmat(self.expInfo.trajectoryPath)
        rawTra=mat['trajectories']
        #take out nan in the beginnin
        nanInd=np.where(np.isnan(rawTra))
        if np.equal(np.shape(nanInd)[1],0) or np.greater(np.max(nanInd),1000):
            LastNan=0
        else:
            LastNan=np.max(nanInd)+1
        
        rawTra=rawTra[LastNan:,:,:]
        self.Pair=Pair(rawTra,self.expInfo)
        
        #generate shifted control 'mock' pairs
        self.sPair=shiftedPair(self.Pair,self.expInfo)
        self.ShoalIndex=(self.sPair.spIAD_m-self.Pair.IAD_m)/self.sPair.spIAD_m
        self.totalPairTravel=sum(self.Pair.totalTravel)
        #Plot results
    
    def plotOverview(self):
        outer_grid = gridspec.GridSpec(4, 3)        
        plt.figure(figsize=(8, 8))   
        #plt.text(0,0.95,path)
        plt.figtext(0,.01,self.expInfo.aviPath)
        
        plt.subplot(4,3,1,rasterized=True)
        plt.cla()
        plt.plot(self.Pair.position[:,0,0],self.Pair.position[:,0,1],'b.',markersize=1,alpha=0.1)
        plt.plot(self.Pair.position[:,1,0],self.Pair.position[:,1,1],'r.',markersize=1,alpha=0.1)
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        plt.title('raw trajectories')     
        
        #plot IAD time series
        x=np.arange(float(np.shape(self.Pair.IAD)[0]))/(self.expInfo.fps*60)
        plt.subplot(4,1,2,rasterized=True)
        plt.cla()
        plt.plot(x,self.Pair.IAD,'b.',markersize=1)
        plt.xlim([0, 90]) 
        plt.xlabel('time [minutes]')
        plt.ylabel('IAD [mm]')
        plt.title('Inter Animal Distance over time')
        
        plt.subplot(4,3,2)
        plt.cla()
        #get rid of nan
        IAD=self.Pair.IAD
        IAD=IAD[~np.isnan(IAD)]
        n, bins, patches = plt.hist(IAD, 30, normed=1, histtype='stepfilled') 
        plt.xlabel('IAD [mm]')
        plt.ylabel('p')
        plt.title('IAD')
        
        plt.subplot(4,3,3)
        plt.cla()
        x=[1,2]
        y=[self.Pair.IAD_m,self.sPair.spIAD_m]
        yerr=[0,self.sPair.spIAD_std]
        plt.bar(x, y, yerr=yerr, width=0.5,color='k')
        lims = plt.ylim()
        plt.ylim([0, lims[1]]) 
        plt.xlim([0.5, 3]) 
        plt.ylabel('[mm]')
        plt.xticks([1.25,2.25],['raw','shift'])
        plt.title('mean IAD')
        
        plt.subplot(4,3,7)
        plt.cla()
        meanPosMat=np.nanmean(self.Pair.neighborMat,axis=2)
        plt.imshow(meanPosMat,interpolation='none', extent=[-31,31,-31,31],clim=[0,100],origin='lower')
        plt.title('mean neighbor position')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        
        plt.subplot(4,3,10)
        plt.cla()
        meanPosMat=self.Pair.neighborMat[:,:,0]
        plt.imshow(meanPosMat,interpolation='none', extent=[-31,31,-31,31],clim=[0,100],origin='lower')
        plt.title('mean neighbor position')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        
        plt.subplot(4,3,11)
        plt.cla()
        meanPosMat=self.Pair.neighborMat[:,:,1]
        plt.imshow(meanPosMat,interpolation='none', extent=[-31,31,-31,31],clim=[0,100],origin='lower')
        plt.title('mean neighbor position')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        

        plt.subplot(4,3,8)
        plt.title('accel=f(pos_n)')
        meanForceMat=np.nanmean(self.Pair.ForceMat,axis=2)
        johPlt.plotMapWithXYprojections(meanForceMat,3,outer_grid[7],31,0.01)
        
        plt.subplot(4,3,9)
        plt.cla()
        plt.bar([1,2],self.Pair.totalTravel,width=0.5,color='k')
        plt.title('total travel')
   
        plt.tight_layout()
        plt.show()
        
        pdfPath=self.expInfo.aviPath+'.pdf'
        with PdfPages(pdfPath) as pdf:
            pdf.savefig()