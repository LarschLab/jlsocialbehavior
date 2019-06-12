# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:34:22 2016

@author: jlarsch
"""

import numpy as np
import subprocess
import os
import cv2
#import tkFileDialog
import functions.gui_circle as gc
import functions.ImageProcessor as ImageProcessor
import models.geometry as geometry
import pickle
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import glob
import sys
def getVideoProperties(aviPath):
    
    #check if videoData has been read already
    head, tail = os.path.split(aviPath)
    head=os.path.normpath(head)
    videoPropsFn=os.path.join(head,'videoProps.pickle')
    
    if np.equal(~os.path.isfile(videoPropsFn),-1):
        #read video metadata via ffprobe and parse output
        #can't use openCV because it reports tbr instead of fps (frames per second)
        cmnd = ['c:/ffmpeg/bin/ffprobe', '-show_format', '-show_streams', '-pretty', '-loglevel', 'quiet', aviPath]
        p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        out = str(out)[3:-10]
        decoder_configuration = {}
        for line in out.split('\\r\\n'):
            if '=' in line:
                key, value = str(line).split('=')
                decoder_configuration[key] = value
        
        #frame rate needs special treatment. calculate from parsed str argument        
        nominator, denominator = decoder_configuration['avg_frame_rate'].split('/')
        decoder_configuration['fps'] = int(float(nominator) / float(denominator))
    
        #save video data for re-use            
        with open(videoPropsFn, 'wb') as f:
            pickle.dump([decoder_configuration], f)
            
    else:
        
        
        with open(videoPropsFn, 'rb') as f:
            decoder_configuration=pickle.load(f)[0]
        #print('re-using VideoProps')
                    
    return decoder_configuration
    

    
    
def get_pixel_scaling(aviPath,forceCorrectPixelScaling=0,forceInput=0,bg_file=''):
    #pixel scaling file will typically reside in parent directory where the raw video file lives
    #forceCorrectPixelScaling=0 - force user iput if no previous data exists
    #forceInput=0 - force user input even if data exists - overwrite
    head, tail = os.path.split(aviPath)
    head=os.path.normpath(head)
    try:
        bg_file=glob.glob(head+'\\dishImage*.jpg')[0]
    except:
        #print 'no background image found for pixel scaling, regenerating...'
        bg_file=getMedVideo(aviPath)[1]
    
    parentDir = os.path.dirname(head)
    scaleFile = os.path.join(parentDir,'bgMed_scale.csv')
    
    if np.equal(~os.path.isfile(scaleFile),-2) or forceCorrectPixelScaling:
        #aviPath = tkFileDialog.askopenfilename(initialdir=parentDir,title='select video to generate median for scale information')
        bg_file=getMedVideo(aviPath,bg_file=bg_file)[1]
#        print bg_file, 'run circleGUI'
        scaleData=gc.get_circle_rois(bg_file,'_scale',forceInput)[0]        
      
    elif forceInput or (np.equal(~os.path.isfile(scaleFile),-1) and  forceCorrectPixelScaling):
        scaleData=np.array(np.loadtxt(scaleFile, skiprows=1,dtype=float))
    else:
        print('no PixelScaling found, using 8 pxPmm')
        return 8

    pxPmm=2*scaleData['circle radius']/scaleData['arena size']
    return pxPmm.values[0]
        
def getAnimalLength(aviPath,frames,coordinates,boxSize=200,threshold=20,invert=False):
    cap = cv2.VideoCapture(aviPath)
    #vp=getVideoProperties(aviPath)
    #videoDims = tuple([int(vp['width']) , int(vp['height'])])
#    print videoDims
    eAll=np.zeros((frames.shape[0],coordinates.shape[1],6))
    for i in range(frames.shape[0]): #use FramesToAvg images to calculate median
        string= str(i)+' out of '+ str(frames.shape[0])+' frames.'
        sys.stdout.write('\r'+string) 
        
        f=frames[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES,f)
        image=cap.read()
        try:
            gray = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)
        except:
            gray=image[1]
        for j in range(coordinates.shape[1]):
            g=gray.copy()
            if np.isnan(coordinates[i,j,0]):
                eAll[i,j,:]=np.nan
                #print i,'nan at frame ',f,j
            else:
                currCenter=geometry.Vector(*coordinates[i,j,:].astype('int'))
                crop=ImageProcessor.crop_zero_pad(g,currCenter,boxSize)
       
                if invert:
                    crop=255-crop
                img_binary = ImageProcessor.to_binary(crop.copy(), threshold,invertMe=False)            
                im_tmp2,contours, hierarchy = cv2.findContours(img_binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_center=geometry.Vector(crop.shape[0]/2,crop.shape[1]/2)
    
                cnt = ImageProcessor.get_contour_containing_point(contours,img_center)
                try:            
                    (x,y),(MA,ma),ori=cv2.minAreaRect(cnt[0])
                    eAll[i,j,0:5]=[x,y,MA,ma,ori]
                    eAll[i,j,0:2]=eAll[i,j,0:2]+currCenter
                    mask=crop.copy()*0
                    cv2.drawContours(mask, cnt[0], -1, (255),-1)
                    eAll[i,j,5]=cv2.mean(crop,mask=mask)[0]
                except:
                    #plt.figure()
                    #plt.imshow(crop)
                    #print cnt
                    #print 'position',currCenter
                    #print i,'problem at frame ',f,j
                    eAll[i,j,:]=np.nan

            
    return eAll      


def getAnimalSize(experiment,needFrames=2000,numFrames=40000,boxSize=200,e2=[]):
    avi_path=experiment.expInfo.aviPath
    head, tail = os.path.split(avi_path)
    sizeFile = os.path.join(head,'animalSize.txt')    
    

    
    if ~np.equal(~os.path.isfile(sizeFile),-2):
        print('determining animalSize from data')
        haveFrames=0
        frames=np.zeros(needFrames).astype('int')
        dist=np.zeros(needFrames)
        
        triedFr=[]
        triedD=[]
        while haveFrames<needFrames:
            tryFrame=np.random.randint(1000,numFrames,1)
            minDist=np.max(np.abs(np.diff(experiment.rawTra[tryFrame,:,:],axis=1)))
            if minDist>boxSize:
                frames[haveFrames]=int(tryFrame)
                dist[haveFrames]=minDist
                haveFrames += 1
            else:
                triedFr.append(tryFrame)
                triedD.append(minDist)
        
        
        tra=experiment.rawTra[frames,:,:]
        if e2!=[]:
            tra[:,0,:]=experiment.rawTra[frames,1,:]

            tra[:,1,:]=e2.rawTra[frames,1,:]
            tra[:,0,0]=tra[:,0,0]+512
            #tra[:,:,1]=512-tra[:,:,1]
            print('using shifted secondAnimal trajectory')
        
        #if (int(experiment.expInfo.videoDims[0])/float(tra.max()))>2:
            
        
        tmp=getAnimalLength(avi_path,frames,tra)
        brightness=np.mean(tmp[:,:,5],axis=0).astype('int')
        MA=np.max(tmp[:,:,2:4],axis=2)
        bins=np.linspace(0,100,101)
        anSize=[np.argmax(np.histogram(MA[:,0],bins=bins)[0]),np.argmax(np.histogram(MA[:,1],bins=bins)[0])]
    
        df=pd.DataFrame({'anID':[1,2],'anSize':anSize,'brightness':brightness},index=None)
        df.to_csv(sizeFile,sep='\t',index=False)
        anID=np.array([1,2])

        ret=np.vstack([anID,anSize]).T     
    else:
#        print 'loading saved animalSize'
        tmp = pd.read_csv(sizeFile, dtype=int, delim_whitespace=True, skipinitialspace=True)
        
        ret = np.array(tmp[[0, 1]].values)  # np.array(np.loadtxt(sizeFile, skiprows=1,dtype=int))
        
    return ret
    
def getMedVideo(aviPath,FramesToAvg=9,saveFile=1,forceInput=0,bg_file='',saveAll=0):
    
    head, tail = os.path.split(aviPath)
    if bg_file=='':
        bg_file=(aviPath[:-4]+'_bgMed.tif')
    
    #print bg_file
    if np.equal(~os.path.isfile(bg_file),-2) and not forceInput:
        bg=cv2.imread(bg_file)
        try:
            bg = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
        except:
            pass
        return bg,bg_file
        
    else:
        print('calculating median video')
        cap = cv2.VideoCapture(aviPath)
        vp=getVideoProperties(aviPath)
        videoDims = tuple([int(vp['width']) , int(vp['height'])])
        print(videoDims)
        #numFrames=int(vp['nb_frames'])
        numFrames=np.min([40000,int(vp['nb_frames'])])
        img1=cap.read()
        img1=cap.read()
        img1=cap.read()
        img1=cap.read()
        gray = cv2.cvtColor(img1[1], cv2.COLOR_BGR2GRAY)
        allMed=gray.copy()
        for i in range(10,numFrames-2,int(np.round(numFrames/FramesToAvg))): #use FramesToAvg images to calculate median
            cap.set(cv2.CAP_PROP_POS_FRAMES,i)
            image=cap.read()
            print(i)
            gray = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)  
            allMed=np.dstack((allMed,gray))
            


        def clip8bit(a):
            a[a>255]=255
            a[a<0]=0
            a[~np.isfinite(a)]=255
            return a.astype('uint8')
            
        def bgDiv8bit(a,b):
            
            tmp=255*(a.astype('float')/b.astype('float'))
            return clip8bit(tmp)
        
        def stretchNorm(x):
            return (x-x.min()).astype('float')/(x.max()-x.min())
            
        def stretchRange(x,mi,ma):
            return (x-mi).astype('float')/(ma-mi)
            
        def norm(x):
            return x.astype('float')/x.max()
            
        def stretchDirect(x,mi,ma):
            xf=x.astype('float')
            mif=float(mi)
            maf=float(ma)
            return clip8bit((xf-mif)*(maf/(maf-mif)))
        
        vidMed=(np.median(allMed,axis=2)).astype('uint8')
        cv2.imwrite(bg_file,vidMed)
        
        if saveAll:
            vidMed=cv2.imread(bg_file)
            vidMed = cv2.cvtColor(vidMed, cv2.COLOR_RGB2GRAY)
            vidMed=np.expand_dims(vidMed,axis=2)
            
            
            allMed_bgSub=bgDiv8bit(allMed,vidMed)
    #        cv2.imshow('allMed_bgSub',allMed_bgSub[:,:,0])        
    #        allMed_bgSub_norm=(norm(allMed_bgSub)*255).astype('uint8')
            
    #        allMed_bgSub_norm=allMed_bgSub.copy()
    #        allMed_bgSub_norm[allMed_bgSub_norm>1]=1
    #        allMed_bgSub_norm=norm(allMed_bgSub.copy())
    #        allMed_bgSub_norm=(allMed_bgSub_norm*255)
            
            #cv2.imshow('allMed_bgSub_norm',allMed_bgSub_norm[:,:,0])
            
            minval2=np.min(allMed_bgSub)-5 #allow for some buffer to the range
    #        minval2=np.min(allMed) 
            #maxval2=np.max(allMed_bgSub)
            
    #        allMedStretch=(stretchRange(allMed,minval2,maxval2)*255).astype('uint8')
            allMedStretch=stretchDirect(allMed,minval2,255)
            #allMedStretch[allMedStretch<1]=1
    #        cv2.imshow('allMedStretch',allMedStretch[:,:,0])
            
    #        vidMedStretch=(np.median(allMedStretch,axis=2)).astype('uint8')
            vidMedStretch=stretchDirect(vidMed,minval2,250)
    #        vidMedStretch=np.expand_dims(vidMedStretch,axis=2)
            
            
    #        vidMedStretch[vidMedStretch<1]=1
    #        cv2.imshow('vidMedStretch',vidMedStretch)
            
            allMedStretch_bgSub=bgDiv8bit(allMedStretch,vidMedStretch)
    #        allMedStretch_bgSub[allMedStretch_bgSub>1]=1
    #        allMedStretch_bgSub=norm(allMedStretch_bgSub)*255
    #        allMedStretch_bgSub[allMedStretch_bgSub<1]=1
            
            print(type(allMedStretch_bgSub))
            
            
            if saveFile:
                
                bg_file=(head+'/bgMed_stretch.tif')
                cv2.imwrite(bg_file,vidMedStretch)
                
                av_file=(head+'/bgcorrect.avi')
                fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
                writer = cv2.VideoWriter(av_file, fourcc, 30, (vidMed.shape[0],vidMed.shape[1]))
    
                for i in range(allMedStretch_bgSub.shape[2]):
                    print(i)
                    x=(np.squeeze(allMedStretch_bgSub[:,:,i])).astype('uint8')
                    writer.write(x)
    
    #            cv2.imshow('allMedStretch_bgSub',x)
                bg_file=(head+'/bgMed_correctedframe.tif')
                cv2.imwrite(bg_file,x)
                writer.release()                                                                                                        

        return vidMed,bg_file