# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:29:56 2016

@author: jlarsch
"""

import legacy.models.AnimalShapeParameters as AnimalShapeParameters
import tkFileDialog
import os
#from tifffile import imsave
import numpy as np
import models.experiment as xp
import matplotlib.pyplot as plt
import models.geometry as geometry
import multiprocessing as mp
import functions.peakdet as peakdet
import pickle
import scipy.stats as sta
import functions.matrixUtilities_joh as mu


if __name__ == '__main__':
    mp.freeze_support()
    reread=1
    numframes=0 #0 = all
    KeepVideoFrames=0; #memory intensive! only for generating example movie...

         
    if reread:
        af = tkFileDialog.askopenfilename(initialdir=os.path.normpath('D:/data/b/2016/20160311_arenaSize/b/1/'))
        head, tail = os.path.split(af)
        head=os.path.normpath(head)
        rotVideoPath = os.path.join(head,tail[:-4]+'_rotateBoth_mask.avi')
        csvPath = os.path.join(head,tail[:-4]+'_contour_orientation.csv')
        
        experiment=xp.experiment(af)
#        pair_trajectories=experiment.pair.get_var_from_all_animals(['ts','rawTra','xy'])
        pair_trajectories=experiment.rawTra
        asp=AnimalShapeParameters.vidSplit(af,pair_trajectories,numframes,KeepVideoFrames)
        
    else:
        pf = tkFileDialog.askopenfilename(initialdir=os.path.normpath('D:/data/b/2016/20160311_arenaSize/b/1/'))
        f=open(pf,'r')
        asp_f=pickle.load(f)
        af=pf[:-22]+'.avi'
        experiment=xp.experiment(af)
        if type(asp_f[0]) is list: #this collection still contains individual animal shape objects for each frame
            asp=[]            
            asp.append(AnimalShapeParameters.AnimalShapeParameters(asp_f,0))
            asp.append(AnimalShapeParameters.AnimalShapeParameters(asp_f,1))
            asp=AnimalShapeParameters.asp_cleanUp(asp)
            
            pickleFile=pf[:-7]+'_+'.pickle
            AnimalShapeParameters.save_asp(pickleFile,asp)
        else:
            asp=asp_f
#b=hhh[0] 
#
            
            
    #collect asp variables for storage
    ori=np.array([a.fish_orientation_elipse for a in asp]).astype('float32')
        
    tailCurvature=np.array([a.allDist.T for a in asp]).astype('float32').T
    tailCurvature_mean=np.nanmean(tailCurvature,axis=1)
    
    centroidContour=np.array([a.centroidContour.T for a in asp]).astype('float32').T
    
    head, tail = os.path.split(af)
    head=os.path.normpath(head)
    aspPath = os.path.join(head,tail[:-4]+'_asp.npz')
    #np.savez('d:/zzsaveTest',tailCurvature_mean=tailCurvature_mean,ori=ori,centroidContour=centroidContour)

    np.savez(aspPath,tailCurvature_mean=tailCurvature_mean,ori=ori,centroidContour=centroidContour)

    plt.figure()
    n, bins, patches =plt.hist(asp[0].deviation, bins=range(-180,180,1), normed=1, histtype='step',lw=3)
    n, bins, patches =plt.hist(asp[1].deviation,bins=range(-180,180,1), normed=1, histtype='step',lw=3)
    plt.title('errorAngle all frames')    
    
#    plt.figure()
#    n, bins, patches =plt.hist(dev0b, bins=range(-180,180,1), normed=1, histtype='step')
#    n, bins, patches =plt.hist(dev1b,bins=range(-180,180,1), normed=1, histtype='step')
#    plt.title('errorAngle all frames')   
#    
#    
    
    animal_dif=asp[0].centroidContour - asp[1].centroidContour
    dist=np.sqrt(animal_dif[:,0]**2 + animal_dif[:,1]**2)
    near=dist<260
    
    
    plt.figure()

    n1, bins, patches =plt.hist(asp[0].deviation[near],bins=range(-180,180,1), normed=1, histtype='step',label='animal 1')
    n2, bins, patches =plt.hist(asp[1].deviation[near],bins=range(-180,180,1), normed=1, histtype='step',label='animal 2')
    plt.title(['errorAngle dist<'+str(int(260/8.6))])
    plt.legend()
    
    
    
    
    # tail beat & bout analysis
    ad=asp[0].allDist.T
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(ad,cmap='hot',clim=[0,10])
    ax.set_aspect('auto')
    ax.grid(False)

    plt.title('tail segment distance to mid line aka tail deflection')
    
    plt.figure()
    tail_curve_1=np.nanmean(asp[0].allDist.T,axis=0)
    plt.plot(tail_curve_1,'-',label='tail curvature') 
    #detect bouts
    bout_start=peakdet.detect_peaks(tail_curve_1[10:-10],1,8)+10
    plt.plot(bout_start,tail_curve_1[bout_start],'o',label='bouts')
    plt.title('tail curvature animal 0 -> peaks = bout start')
    plt.legend()
#    #for comparison: change in heading - should coincide with tail beats
#    plt.figure()
#    d_head_sm=geometry.smallest_angle_difference_degrees(asp[0].fish_orientation_elipse[:-1],asp[0].fish_orientation_elipse[1:])
#    d_head_sm=np.abs(d_head_sm)
#    d_heading_norm=d_head_sm/np.nanmax(d_head_sm)*np.nanmax(tail_curve_1)
#    plt.plot(d_heading_norm,'-') 
#
#    plt.figure()
#    plt.hist(d_head_sm,bins=range(180))

    


    
    ##bout analysis
    #compare error angle before vs after each bout
    #(error angle is called 'deviation' in asp class)
    
    i=0
    error_before=np.nanmedian([asp[0].deviation[b-6:b-3] for b in bout_start],axis=1)
    error_after=np.nanmedian([asp[0].deviation[b+3:b+6] for b in bout_start],axis=1)
    d_error=np.diff(np.abs([error_before,error_after]).T,axis=1)

    ##alternative d_error 
    #considering instantaneous positions of both animals during the bout.    
    asp=AnimalShapeParameters.asp_cleanUp(asp)
    #baseline is orientation of vector from ainimal 0 to animal 1
    baseline1=[asp[0].baseline[b] for b in bout_start]
    #heading is median animal orientation
    heading_before=np.nanmedian([asp[0].fish_orientation_elipse[b-7:b-2] for b in bout_start],axis=1)
    heading_after=np.nanmedian([asp[0].fish_orientation_elipse[b+2:b+7] for b in bout_start],axis=1)
    #error(angle) is difference baseline-heading    
    error_before=geometry.smallest_angle_difference_degrees(heading_before,baseline1) 
    error_after=geometry.smallest_angle_difference_degrees(heading_after,baseline1) 
    d_error_inst=np.diff(np.abs([error_before,error_after]).T,axis=1)
    d_error_inst2=np.diff(np.array([error_before,error_after]).T)
    d_heading=geometry.smallest_angle_difference_degrees(heading_before,heading_after)
    
    #change in distance
    d_IAD=np.diff(dist)    
    plt.figure()
    plt.plot(d_IAD)
    
    plt.figure()
    plt.scatter(dist[1:],d_IAD)
    
    d_IAD_lastBout=[dist[b]-dist[b-1] for b in bout_start[1:]]
    plt.figure()
    plt.plot(dist)
    plt.plot(bout_start[1:],np.multiply(d_IAD_lastBout,50),'.')
    
    
    plt.figure()
    plt.scatter(d_IAD_lastBout,d_heading[1:])    
    
    
    plt.figure()
    binsx=np.arange(-20,20,.5)
    binsy=np.arange(0,180,5)
    tmp=plt.hist2d(d_IAD_lastBout,np.abs(error_before[1:]),bins=[binsx,binsy])[0]
    bm=sta.binned_statistic(d_IAD_lastBout,np.abs(error_before[1:]),bins=binsx,statistic='median')[0]
    plt.plot(binsx[1:],bm)
    
    
    plt.figure()
    binsx=np.arange(-20,20,.5)
    binsy=np.arange(0,180,1)
    tmp=plt.hist2d(d_IAD_lastBout,np.abs(error_after[1:]),bins=[binsx,binsy])[0]
    bm=sta.binned_statistic(d_IAD_lastBout,np.abs(error_after[1:]),bins=binsx,statistic='median')[0]
    plt.plot(binsx[1:],bm)
    
    plt.figure()
    binsx=np.arange(-20,20,1)
    binsy=np.arange(-25,25,1)
    tmp=plt.hist2d(d_IAD_lastBout,np.squeeze((d_error_inst[1:,0])),bins=[binsx,binsy])[0]
    bm=sta.binned_statistic(np.array(d_IAD_lastBout),np.squeeze((d_error_inst[1:,0])),bins=binsx,statistic='mean')[0]
    plt.plot(binsx[1:],bm)
    
    
    plt.figure()
    binsx=np.arange(-180,180,5)
    binsy=np.arange(-45,45,1)
    edhMap=plt.hist2d(error_before,np.squeeze((d_error_inst)),bins=[binsx,binsy])[0]
    m=np.isfinite(error_before)==True & np.isfinite(d_heading)
    #sta.pearsonr(error_before[m],d_heading[m])
 
    bm=sta.binned_statistic(error_before[m],np.squeeze((d_error_inst))[m],bins=binsx,statistic='mean')[0]
    plt.plot(binsx[1:],bm)
    

    plt.scatter(error_before,d_heading)
    binsx=np.arange(-180,180,5)
    binsy=np.arange(-90,90,5)
    plt.figure()
    edhMap=plt.hist2d(error_before,d_heading,bins=[binsx,binsy])[0]
    m=np.isfinite(error_before)==True & np.isfinite(d_heading)
    #sta.pearsonr(error_before[m],d_heading[m])
 
    bm=sta.binned_statistic(error_before[m],d_heading[m],bins=binsx,statistic='mean')[0]
    plt.plot(binsx[1:],bm)
     
    
    heading_before=np.nanmedian([asp[0].fish_orientation_elipse[b-7:b-2] for b in bout_start],axis=1)
    heading_after=np.nanmedian([asp[0].fish_orientation_elipse[b+2:b+7] for b in bout_start],axis=1)
    #error(angle) is difference baseline-heading    
    error_before=geometry.smallest_angle_difference_degrees(heading_before,baseline1) 
    error_after=geometry.smallest_angle_difference_degrees(heading_after,baseline1) 
    d_error_inst=np.diff(np.abs([error_before,error_after]).T,axis=1)
    d_error_inst2=np.diff(np.array([error_before,error_after]).T)
    d_heading=geometry.smallest_angle_difference_degrees(heading_before,heading_after)
    
    plt.figure()
    plt.scatter(error_before,d_heading)
    binsx=np.arange(-180,180,5)
    binsy=np.arange(-90,90,5)
    plt.figure()
    edhMap=plt.hist2d(error_before,d_heading,bins=[binsx,binsy])[0]
    m=(np.isfinite(error_before)==True) & (np.isfinite(d_heading)==True)
    #sta.pearsonr(error_before[m],d_heading[m])
 
    bm=sta.binned_statistic(error_before[m],d_heading[m],bins=binsx,statistic='mean')[0]
    plt.plot(binsx[1:],bm)
    
    
    
    plt.figure()
    n1, bins, patches =plt.hist(error_before,bins=range(-180,180,5), normed=1, histtype='step',label='before bout')    
    n1, bins, patches =plt.hist(error_after,bins=range(-180,180,5), normed=1, histtype='step',label='after bout')    
    plt.title('animal 0 error angle before vs after bout (all frames)')
    plt.legend()

    dist_at_bout=dist[bout_start]
    near=dist_at_bout<260
    
    plt.figure()
    n1, bins, patches =plt.hist(error_before[near],bins=range(-180,180,5), normed=1, histtype='step')    
    n1, bins, patches =plt.hist(error_after[near],bins=range(-180,180,5), normed=1, histtype='step')    
    plt.title('animal 0 error angle before vs after bout (dist<260)')

    
    plt.figure()   
    n1, bins, patches =plt.hist(d_error_inst,bins=range(-100,100,1), normed=1, histtype='step')    
    plt.plot([0,0], [0, np.max(n1)], 'k:', lw=3)
    plt.title('animal 0 change in error angle before vs after bout (instantaneous, all frames)')


    plt.figure()   
    n1, bins, patches =plt.hist(d_error,bins=range(-100,100,1), normed=1, histtype='step')    
    plt.plot([0,0], [0, np.max(n1)], 'k:', lw=3)
    plt.title('animal 0 change in error angle before vs after bout (target moving, all frames)')


    #is bout-directed-ness modulated with animal distance?    
    dist_at_bout=dist[bout_start[1:-2]]
        
    plt.figure()
    plt.scatter(dist_at_bout,d_error_inst)
    plt.title('change in error angle over distance (instantaneous, all frames)')

    
    
    #variable binning - spaced to fill each bin with similar number of data points
    mapBinsx=(np.sort(dist_at_bout)[::int(dist_at_bout.shape[0]/180)]).astype('int')


    bm=sta.binned_statistic(dist_at_bout,d_error_inst[:,0],bins=mapBinsx,statistic='mean')[0]
    plt.figure()
    plt.plot(mapBinsx[:-1]+np.diff(mapBinsx)/2,bm,'.')
    plt.plot([mapBinsx[0],mapBinsx[-1]], [0, 0], 'k:', lw=3)
    plt.title('animal 0 change in error angle -binned mean - variable bins')

    #plot this data in heat map style
    #effectively a map of histogramms for each distance bin of the data
    #normalize each column by # of data points
    #mapBinsx=np.arange(0,1500,2)
    mapBinsy=np.arange(-120,120,20)

    plt.figure()
    plt.hist2d(dist_at_bout,np.squeeze(d_error_inst),bins=[mapBinsx,mapBinsy])
    plt.plot([mapBinsx[0],mapBinsx[-1]], [0, 0], 'k:', lw=3)
    
    def movingaverage(interval, window_size):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')
    
    bm_av=movingaverage(bm,3)
    plt.plot(mapBinsx[:-1]+np.diff(mapBinsx)/2,bm_av,'r', lw=4)
    plt.ylim([-50,50])
    plt.xlim([mapBinsx[0],mapBinsx[-1]])
    plt.title('animal 0 change in error angle -binned mean - variable bins')

    
    #is bout interval modulated with animal distance? -->seems not!    
    bout_interval=np.diff(bout_start)
    dist_at_bout=dist[bout_start]
        
    plt.figure()
    plt.scatter(dist_at_bout[:-1],bout_interval)
    plt.title('bout interval over animal distance')


    #mapBins=np.arange(0,2000,50)
    mapBinsx=(np.sort(dist_at_bout)[::int(dist_at_bout.shape[0]/180)]).astype('int')

    bm=sta.binned_statistic(dist_at_bout[:-1],bout_interval,bins=mapBinsx)[0]
    plt.figure()
    plt.plot(mapBinsx[:-1]+np.diff(mapBinsx)/2,bm,':.')
    plt.title('bout interval over animal distance - binned mean')

    #heat map - not useful, needs bin approach analogous to above if picked up later
#    mapBinsy=np.arange(0,100,5)
#    
#    bma=np.zeros((mapBinsx.shape[0]-1,mapBinsy.shape[0]-1))
#
#    for i in range(mapBinsx.shape[0]-1):
#        mask=np.multiply(dist_at_bout[:-1]>mapBinsx[i] , dist_at_bout[:-1]<mapBinsx[i+1])
#        bma[i,:]=np.histogram(bout_interval[mask],bins=mapBinsy,normed=1)[0]
#    plt.figure()
#    plt.imshow(bma.T)
    
    #is bout interval modulated with relative orientation? (only considering 'close animals')
    #--> maybe
    dist_at_bout=dist[bout_start]
    bout_interval=np.diff(bout_start)
    dev_at_bout=asp[0].deviation[bout_start]
    distLim=260
    distMask=dist_at_bout<distLim
    dev_at_bout=dev_at_bout[distMask]
    bout_interval=bout_interval[distMask[:-1]]
        
    plt.figure()
    plt.scatter(dev_at_bout,bout_interval)
    
    #mapBinsx=np.arange(-180,180,20)
    mapBinsx=(np.sort(dev_at_bout)[::int(dev_at_bout.shape[0]/40)]).astype('int')

    mapBinsy=np.arange(0,40,1)

    bm=sta.binned_statistic(dev_at_bout,bout_interval,bins=mapBinsx,statistic='median')[0]
    plt.figure()
    plt.plot(mapBinsx[:-1]+np.diff(mapBinsx)/2,bm,'.:')
    plt.title('animal 0 bout interval over error angle - binned median')


    
    bma=np.zeros((mapBinsx.shape[0]-1,mapBinsy.shape[0]-1))

    for i in range(mapBinsx.shape[0]-1):
        mask1=np.multiply(dev_at_bout[:-1]>mapBinsx[i] , dev_at_bout[:-1]<mapBinsx[i+1])
        
        bma[i,:]=np.histogram(bout_interval[mask1],bins=mapBinsy,normed=1)[0]
    plt.figure()
    plt.imshow(bma.T)
    
    
    #is bout travel modulated with relative orientation? (only considering 'close animals')
    
    a1=np.diff(experiment.Pair.animals[0].position.xy[bout_start,:],axis=0)
    t1=mu.distance(*a1.T)
    

    
    dist_at_bout=dist[bout_start]
    dev_at_bout=asp[0].deviation[bout_start]
    distLim=260
    distMask=dist_at_bout<distLim
    #dev_at_bout=dev_at_bout[distMask[:-1]]
    #t1=t1[distMask]
    dev_at_bout=dev_at_bout[:-1]
    dev_at_bout=np.mod(dev_at_bout+360,360)-180 #this is fudge, check!!!!

    #mapBinsx=np.arange(-180,180,20)
    mapBinsx=(np.sort(dev_at_bout)[::int(dev_at_bout.shape[0]/40)]).astype('int')

    mapBinsy=np.arange(0,40,1)

    bm=sta.binned_statistic(dev_at_bout,t1,bins=mapBinsx,statistic='mean')[0]
    plt.figure()
    plt.plot(mapBinsx[:-1]+np.diff(mapBinsx)/2,bm,'.:',label='animal 1')
    plt.title('bout travel over error angle - binned mean')
    
    tail_curve_2=np.nanmean(asp[1].allDist.T,axis=0)
    #detect bouts
    bout_start2=peakdet.detect_peaks(tail_curve_2,1,8)
    a2=np.diff(experiment.Pair.animals[1].position.xy[bout_start2,:],axis=0)
    t2=mu.distance(*a2.T)
    dist_at_bout2=dist[bout_start2]
    dev_at_bout2=asp[1].deviation[bout_start2]
    dev_at_bout2=dev_at_bout2[:-1]
    mapBinsx=(np.sort(dev_at_bout2)[::int(dev_at_bout2.shape[0]/40)]).astype('int')
    mapBinsy=np.arange(0,40,1)
    bm=sta.binned_statistic(dev_at_bout2,t2,bins=mapBinsx,statistic='mean')[0]
    plt.plot(mapBinsx[:-1]+np.diff(mapBinsx)/2,bm,'.:',label='animal 2')
    plt.legend()

    #
#    
#
##contour_orientation = np.array([a, b]).T
##contour_orientation_rad=np.deg2rad(contour_orientation-180)
##np.savetxt(csvPath,contour_orientation)
##    
#angle_diff=geometry.smallest_angle_difference_degrees(a,b)
#
#angle_diff_shift=[]
#for i in range(10):
#    bs=b.copy()
#    minShift=5*6*30
#    shiftIndex=int(random.uniform(minShift,bs.shape[0]-minShift))
#    bs=np.roll(bs,shiftIndex)
#    angle_diff_shift.append(geometry.smallest_angle_difference_degrees(a,bs))
#
#    
#    aa=experiment.Pair.heading[:,0,0]
#    bb=experiment.Pair.heading[:,1,0]
#    
#    #plt.plot(asp1.fish_orientation_elipse_all[1:4000])
#    #plt.plot(asp2.fish_orientation_elipse_all[1:4000])
#    #plt.plot(np.degrees(aa)+180)
#    #plt.scatter(np.degrees(aa[:-1])+180,asp1.fish_orientation_elipse_all[1:])
#    ac=a.copy()
#    ac[np.isnan(ac)]=0
#    bc=b.copy()
#    bc[np.isnan(bc)]=0
#    c=np.correlate(bc,ac,mode='full')
#    plt.figure()
#    plt.plot(c)
#    c=np.correlate(ac,bc,mode='full')
#    plt.plot(c)
#    plt.figure()
#    c=np.correlate(bb,aa,mode='full')
#    plt.plot(c)
#    c=np.correlate(aa,bb,mode='full')
#    plt.plot(c)
#
#chunkLength=60*30
#chunk_lag=[]
#chunk_lead=[]
#chunk_IADm=[]
#for i in range(2,np.int((a.shape[0])/chunkLength)-2):
#    c_range1=range((i-1)*chunkLength,(i+2)*chunkLength-1)
#    c_range2=range(i*chunkLength,(i+1)*chunkLength-1)
#    c_corr=np.correlate(bc[c_range1],ac[c_range2],mode='same')
#    c_lag=np.argmax(c_corr)-np.shape(c_range2)[0]
#    chunk_lag.append(c_lag)
#    c_neighborMat,c_relPosRot,c_relPos=joFishHelper.getRelativeNeighborPositions(experiment.Pair.position[c_range2,:,:],contour_orientation_rad[c_range2[1:]])
#    c_lead=joFishHelper.leadership(c_neighborMat)
#    chunk_lead.append(c_lead)
#    chunk_IADm.append(np.nanmean(experiment.Pair.IAD[c_range2]))
#    
#leadReal=joFishHelper.leadership(experiment.Pair.neighborMat)
#lead_shifted=[]
#lead_shifted.append([joFishHelper.leadership(y) for y in ([x.neighborMat for x in experiment.sPair.sPair])])
#
#
##plt.scatter(np.array(chunk_lead)[:,0],np.array(chunk_lag))
##plt.plot([0, 0], [-300, 300], 'k:', lw=1)
##plt.plot([-1, 1], [0, 0], 'k:', lw=1)
##plt.figure()
##plt.plot(chunk_lag,'o')
#plt.figure()
#plt.plot(chunk_lead)
#
#plt.figure()
#plt.plot(chunk_IADm)
#