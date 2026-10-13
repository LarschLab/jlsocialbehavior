# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:29:56 2016

@author: jlarsch
"""

import legacy.models.AnimalShapeParameters as asp
import tkFileDialog
import os
#from tifffile import imsave
import numpy as np
import models.experiment as xp
import matplotlib.pyplot as plt
import models.geometry as geometry
import multiprocessing as mp
import functions.peakdet as peakdet



#b=hhh[0] 
#
#plt.figure()
#n, bins, patches =plt.hist(asp[0].deviation,bins=range(-180,180,10), normed=1, histtype='step')
#n, bins, patches =plt.hist(asp[1].deviation,bins=range(-180,180,10), normed=1, histtype='step')
#
#
#ad=np.array([x.allDist.T for x in asp])
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.imshow(ad)
#ax.set_aspect('auto')
#
#plt.figure()
#tail_curve_1=np.nanmean(asp[1].allDist.T,axis=0)
#plt.plot(tail_curve_1,'o-')
#
##detect bouts
#
#bout_start=peakdet.detect_peaks(tail_curve_1,1,8)

#plt.plot(xxx,np.abs(d_orientation_b)[xxx],'o')
#plt.figure()
#plt.plot(np.mod(np.abs(d_orientation_b),180))
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



if __name__ == '__main__':
    mp.freeze_support()
    reread=1

    af = tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b/2016/20160314_TLStacks/1/a/1/'))
    head, tail = os.path.split(af)
    head=os.path.normpath(head)
    rotVideoPath = os.path.join(head,tail[:-4]+'_rotateBoth_mask.avi')
    csvPath = os.path.join(head,tail[:-4]+'_contour_orientation.csv')
    
    experiment=xp.experiment(af)
    pair_trajectories=experiment.Pair.get_var_from_all_animals(['rawTra','xy'])
    
    #    asp2=AnimalShapeParameters.AnimalShapeParameters(af,t[:,1,:],pair_trajectories.shape[0])
    #    asp1=AnimalShapeParameters.AnimalShapeParameters(af,t[:,0,:],pair_trajectories.shape[0])
    
    
              
                
    if reread:
        #asp=asp.get_AnimalShapeParameters(af,pair_trajectories,pair_trajectories.shape[0])
    #    asp=asp.get_AnimalShapeParameters(af,pair_trajectories,100)
        asp=asp.vidSplit(af,pair_trajectories)

