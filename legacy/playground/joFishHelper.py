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
import randSpacing
import cv2
from matplotlib.backends.backend_pdf import PdfPages
import geometry
import ImageProcessor

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
        
        trajectoryPath = os.path.join(head,'trajectories_nogaps.mat')
        if np.equal(~os.path.isfile(trajectoryPath),-2):
            self.trajectoryPath = trajectoryPath
        else:
            trajectoryPath = os.path.join(head,'trajectories.mat')
            if np.equal(~os.path.isfile(trajectoryPath),-2):
                self.trajectoryPath = trajectoryPath
                
        AnSizeFilePath = os.path.join(head,'animalSize.txt')
        if np.equal(~os.path.isfile(AnSizeFilePath),-2):
            self.AnSizeFilePath = AnSizeFilePath
        
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
        expInfo.trajectoryDiameterPx=np.mean(maxPixel-minPixel)
        #expInfo.pxPmm=expInfo.trajectoryDiameterPx/expInfo.arenaDiameter_mm
        expInfo.pxPmm=8.6
        expInfo.arenaCenterPx=np.mean(maxPixel-(expInfo.trajectoryDiameterPx/2),axis=0)
        
        self.position=(self.positionPx-expInfo.arenaCenterPx) / expInfo.pxPmm
        
        #distance from center
        self.positionPol=np.transpose(mu.cart2pol(self.position[:,:,0],self.position[:,:,1]),[1,2,0])  
        maxCenterDistance=np.max(self.positionPol[:,:,1])
        histBins=np.linspace(0,maxCenterDistance,100)   
        self.PolhistBins=histBins
        self.Pol_n=np.zeros([99,2])
        histData=self.positionPol[:,0,1]
        self.Pol_n[:,0],bins =np.histogram(histData[~np.isnan(histData)],bins=histBins,normed=1)
        histData=self.positionPol[:,1,1]
        self.Pol_n[:,1],bins =np.histogram(histData[~np.isnan(histData)],bins=histBins,normed=1)
        if self.Pol_n[-2,0]==0: #second last bin of animal0 is empty, meaning animal1 went out further ->was on top
            self.StackTopAnimal=np.array([0,1])
        elif self.Pol_n[-2,1]==0:
            self.StackTopAnimal=np.array([1,0])
        else: #no animal went out more than one bin further than the other -> likely no stack experiment
            self.StackTopAnimal=np.array([0,0])       
        
        #change in position for x and y
        self.d_position=np.diff(self.position,axis=0)
        self.dd_position=np.diff(self.d_position,axis=0)
        
        #travel distance (cartesian displacement)
        self.travel=np.sqrt(self.d_position[:,:,0]**2 + self.d_position[:,:,1]**2)
        
        #tavel speed
        self.speed=self.travel*expInfo.fps
        self.totalTravel=np.nansum(np.abs(self.travel),axis=0)
        
        #travel acceleration
        self.accel=np.diff(self.speed,axis=0)
        self.heading=np.transpose(mu.cart2pol(self.d_position[:,:,0],self.d_position[:,:,1]),[1,2,0]) #heading[0] = heading, heading[1] = speed
        #heading: pointing right = 0, pointung up = pi/2      
        self.d_heading=np.diff(self.heading[0],axis=0)
        
        #absolute inter animal distance IAD
        
        dist=self.position[:,0,:]-self.position[:,1,:]
        self.IAD = np.sqrt(dist[:,0]**2 + dist[:,1]**2)
        self.IAD_m=np.nanmean(self.IAD)
        histBins=np.arange(100)
        self.IADhist,bins =np.histogram(self.IAD,bins=histBins,normed=1)
        
        #relative distance between animals
        self.neighborMat,self.relPosPolRotCart,self.relPos= getRelativeNeighborPositions(self.position,self.heading)
        
        #force map between animals
        self.ForceMat = getNeighborForce(self.relPosPolRotCart,self.accel)
        
        #self.dwellH,self.dwellL,self.dwellHTL,self.dwellLTH=getShoalDwellTimes(self.IAD)

        
def getRelativeNeighborPositions(position,heading):
    if heading.ndim==3:
        hd=heading[:,:,0].copy()
    elif heading.ndim==2:
        hd=heading[:,:].copy()
        
    pos=position[1:,:,:].copy() #obtain a new copy rather than using reference
    
    relPos1=pos[:,0,:]-pos[:,1,:] #position of animal 1 relative to animal 2. (where animal 2 has a neighbor)
    relPos2=pos[:,1,:]-pos[:,0,:]  
    
    relPos=np.transpose([relPos1,relPos2],[1,0,2])
    
    relPosPol=np.transpose(mu.cart2pol(relPos[:,:,0],relPos[:,:,1]),[1,2,0])
    relPosPolRot=relPosPol.copy()
    #rotate 
    relPosPolRot[:,0,0]=relPosPol[:,0,0]-(hd[:,1])
    relPosPolRot[:,1,0]=relPosPol[:,1,0]-(hd[:,0])
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



def distanceTimeSeries(X):
    if ~('result' in locals()):
        result=np.array([])
    
    result=np.append(result,abs(X[0]-X[1:]))
        
    if np.shape(X)[0]>2:
        result=np.append(result,distanceTimeSeries(X[1:]))

    return result

class shiftedPair(object):
    #Class to calculate null hypothesis time series using shifted pairs of animals
    def __init__(self, tra,expInfo):
        self.nRuns=10
        self.minShift=5*60*expInfo.fps
        self.sPair=[]
        #generate nRuns instances of Pair class with one animal time shifted against the other
        for i in range(self.nRuns):
            traShift=tra.positionPx.copy()
            shiftIndex=int(random.uniform(self.minShift,traShift.shape[0]-self.minShift))
            #time-rotate animal 0, keep animal 1 as is
            traShift[:,0,:]=np.roll(traShift[:,0,:],shiftIndex,axis=0)
            self.sPair.append(Pair(traShift,expInfo))
        #calculate mean and std IAD for the goup of shifted instances
        self.spIAD_m = np.nanmean([x.IAD_m for x in self.sPair])
        self.spIAD_std = np.nanstd([x.IAD_m for x in self.sPair])

def shiftedPairSystematic(tra,expInfo,nRuns):
    #function to test systematically the effect of the extent of the time shift
    shiftInterval=expInfo.fps
    sIADList=[]
    #generate nRuns instances of Pair class with one animal time shifted against the other
    for i in range(nRuns):
        traShift=tra.positionPx.copy()
        shiftIndex=i*shiftInterval
        #time-rotate animal 0, keep animal 1 as is
        traShift[:,0,:]=np.roll(traShift[:,0,:],shiftIndex,axis=0)
        tmpPair=Pair(traShift,expInfo)
        sIADList.append(tmpPair.IAD_m)
    return sIADList
    
    

            
    
class xy_point(object):
    def __init__(self,coordinates):
        self.x=coordinates[0]
        self.y=coordinates[1]

def leadership(PosMat):
    front=np.sum(np.sum(PosMat[:31,:,:],axis=1),axis=0)
    back =np.sum(np.sum(PosMat[32:,:,:],axis=1),axis=0)
    LeadershipIndex=(front-back)/(front+back)
    return LeadershipIndex
        
class experiment(object):
    #Class to collect, store and plot data belonging to one experiment
    def __init__(self,path):
        self.expInfo=ExperimentMeta(path)
        self.AnSize=np.array(np.loadtxt(self.expInfo.AnSizeFilePath, skiprows=1,dtype=int))
        mat=scipy.io.loadmat(self.expInfo.trajectoryPath)
        rawTra=mat['trajectories']
        #take out nan in the beginnin DONT do this for now, this would shift trace averages or require correction!
        nanInd=np.where(np.isnan(rawTra))
        if np.equal(np.shape(nanInd)[1],0) or np.greater(np.max(nanInd),1000):
            LastNan=0
        else:
            LastNan=np.max(nanInd)+1
        
        #rawTra=rawTra[LastNan:,:,:]
        self.skipNanInd=LastNan
        self.rawTra=rawTra
        
        #proceed with animal-pair analysis if there is more than one trajectory
        if np.shape(rawTra)[1]>1:
            self.Pair=Pair(rawTra.copy(),self.expInfo)
        
            #self.AnShape=AnimalShapeParameters()
            
            #generate shifted control 'mock' pairs
            self.sPair=shiftedPair(self.Pair,self.expInfo)
            
            #calculate mean IAD histogram for shifted pairs
            IADHistall=[]
            IADHistall.append([x.IADhist[0:30*60*90] for x in self.sPair.sPair])
            self.spIADhist_m=np.nanmean(IADHistall,axis=1)            
            
            self.ShoalIndex=(self.sPair.spIAD_m-self.Pair.IAD_m)/self.sPair.spIAD_m
            self.totalPairTravel=sum(self.Pair.totalTravel)
            self.avgSpeed=np.nanmean(self.Pair.speed)
            probTra=mat['probtrajectories']
            self.idQuality=np.mean(probTra[probTra>=0])*100
            
    
    def plotOverview(self,condition='notDefined'):
        outer_grid = gridspec.GridSpec(4, 4)        
        plt.figure(figsize=(8, 8))   
        #plt.text(0,0.95,path)
        plt.figtext(0,.01,self.expInfo.aviPath)
        plt.figtext(0,.03,condition)
        plt.figtext(0,.05,self.idQuality)
        
        plt.subplot(4,4,1,rasterized=True)
        plt.cla()
        plt.plot(self.Pair.position[:,0,0],self.Pair.position[:,0,1],'b.',markersize=1,alpha=0.1)
        plt.plot(self.Pair.position[:,1,0],self.Pair.position[:,1,1],'r.',markersize=1,alpha=0.1)
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        plt.title('raw trajectories')  
        
        plt.subplot(4,4,3)
        plt.cla()
        plt.step(self.Pair.PolhistBins[:-1], self.Pair.Pol_n[:,0],'b',lw=2,where='mid')
        plt.step(self.Pair.PolhistBins[:-1], self.Pair.Pol_n[:,1],'r',lw=2,where='mid')
        plt.ylim([0, .1])

        
        
        #plot IAD time series
        x=np.arange(float(np.shape(self.Pair.IAD)[0]))/(self.expInfo.fps*60)
        plt.subplot(4,1,2,rasterized=True)
        plt.cla()
        plt.plot(x,self.Pair.IAD,'b.',markersize=1)
        plt.xlim([0, 90]) 
        plt.xlabel('time [minutes]')
        plt.ylabel('IAD [mm]')
        plt.title('Inter Animal Distance over time')
        
        #IAD histogram for raw and shifted data and simulated random dots
        plt.subplot(4,4,2)
        plt.cla()
        #get rid of nan
        IAD=self.Pair.IAD
        IAD=IAD[~np.isnan(IAD)]
        histBins=np.arange(100)
        n, bins, patches = plt.hist(IAD, bins=histBins, normed=1, histtype='stepfilled')
        plt.step(histBins[:-1],self.spIADhist_m.T,'k',lw=1)
        #simulate random uniform spacing
        rad=(self.expInfo.trajectoryDiameterPx/self.expInfo.pxPmm)/2
        num=1000
        dotList,dotND = randSpacing.randomDotsOnCircle(rad,num)
        n, bins, patches = plt.hist(dotND, bins=histBins, normed=1, histtype='step')
        plt.xlabel('IAD [mm]')
        plt.ylabel('p')
        plt.title('IAD')
        plt.ylim([0, .05])
        
        plt.subplot(4,4,4)
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
        
        plt.subplot(4,4,9)
        plt.cla()
        meanPosMat=np.nanmean(self.Pair.neighborMat,axis=2)
        plt.imshow(meanPosMat,interpolation='none', extent=[-31,31,-31,31],clim=[0,100],origin='lower')
        plt.title('mean neighbor position')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        
        #neighborhood matrix for animal0
        #the vertial orientation was confirmed correct in march 2016
        plt.subplot(4,4,13)
        plt.cla()
        meanPosMat=self.Pair.neighborMat[:,:,0]
        plt.imshow(meanPosMat,interpolation='none', extent=[-31,31,-31,31],clim=[0,200],origin='lower')
        plt.title('mean neighbor position')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        
        plt.subplot(4,4,14)
        plt.cla()
        meanPosMat=self.Pair.neighborMat[:,:,1]
        plt.imshow(meanPosMat,interpolation='none', extent=[-31,31,-31,31],clim=[0,200],origin='lower')
        plt.title('mean neighbor position')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        
        #LEADERSHIP
        plt.subplot(4,4,15)
        plt.cla()
        self.LeadershipIndex=leadership(self.Pair.neighborMat)
        x=[1,2]
        barlist=plt.bar(x,self.LeadershipIndex, width=0.5,color='b')
        barlist[1].set_color('r')
        plt.title('Leadership')
        plt.ylabel('index')
        plt.ylim([-.5, .5])
        plt.xlim([0.5, 3])
        
        if np.sum(self.Pair.StackTopAnimal)>0:
            xtickList=[int(np.where(self.Pair.StackTopAnimal==1)[0])+1.25,int(np.where(self.Pair.StackTopAnimal==0)[0])+1.25]
            plt.xticks(xtickList,['top','bottom'])
        else:
            plt.xticks([1.25,2.25],['same','dish'])
            
        plt.subplot(4,4,16)
        plt.cla()
        x=[1,2]
        barlist=plt.bar(x,self.AnSize[:,1], width=0.5,color='b')
        barlist[1].set_color('r')
        plt.xlim([0.5, 3])
        plt.title('Body size')
        plt.ylabel('area [px]')
        


        plt.subplot(4,4,10)
        plt.title('accel=f(pos_n)')
        meanForceMat=np.nanmean(self.Pair.ForceMat,axis=2)
        johPlt.plotMapWithXYprojections(meanForceMat,3,outer_grid[9],31,0.01)
        
        plt.subplot(4,4,11)
        plt.cla()
        plt.bar([1,2],np.nanmean(self.Pair.speed,axis=0),width=0.5)
        plt.title('avgSpeed')
   
        plt.tight_layout()
        plt.show()
        
        pdfPath=self.expInfo.aviPath+'.pdf'
        with PdfPages(pdfPath) as pdf:
            pdf.savefig()