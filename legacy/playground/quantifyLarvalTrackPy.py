# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:24:41 2016

@author: jlarsch
"""


import joFishHelper
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import tkFileDialog
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import numpy.matlib

import pims
import trackpy as tp
import pickle

avi_path = tkFileDialog.askopenfilename(initialdir='d:/data/b/2016/')

FramesToProcess=range(60*60*11,int(60*60*11.25))
nFramesToProcess=np.shape(FramesToProcess)[0]


import cv2
cap = cv2.VideoCapture(avi_path)
keyAll=[]

outFileBase='d:/outData_d'
hdfFile=outFileBase+'.h5'
pickleFile=outFileBase+'.pickle'


with tp.PandasHDFStore(hdfFile) as s:  # This opens an HDF5 file. Data will be stored and retrieved by frame number.
    for i in range(nFramesToProcess):
        f=FramesToProcess[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES,f)
        currImg=cap.read()[1]
        gray=cv2.cvtColor(currImg, cv2.COLOR_BGR2GRAY)
        #small = cv2.resize(gray, (0,0), fx=0.25, fy=0.25)
        fe = tp.locate(gray, 5, invert=True,max_iterations=0)
        fe['frame']=f
        s.put(fe)
        print f,"         \r",
        #tp.annotate(f, gray)
        #plt.draw()

        
with tp.PandasHDFStore(hdfFile) as s:
    for linked in tp.link_df_iter(s, 15, neighbor_strategy='KDTree'):
        s.put(linked)

with tp.PandasHDFStore(hdfFile) as s:
    all_results = s.dump()

condition2 = lambda x: ((x['mass'].mean() > 200))
t3 = tp.filter(all_results, condition2)

#plt.ion()
#plt.figure()
#plt.axis([0,520,0,520])
#
#for f in range(50):
#    i=FramesToProcess[f+70*60]
#    cap.set(cv2.CAP_PROP_POS_FRAMES,i)
#    currImg=cap.read()[1]
#    gray=cv2.cvtColor(currImg, cv2.COLOR_BGR2GRAY)
#    tp.annotate(t3[(t3['frame'] == i) & (t3['particle'] == 11570)], gray)
#    plt.show()
#    plt.pause(0.0001)


numParticles=int(np.max(t3['particle']))
allT=np.zeros([nFramesToProcess+1,130,2])
allT[:]=np.nan
ofs=FramesToProcess[0]
for i in range(numParticles):
    #get current particle
    pc=all_results.loc[all_results['particle']==i]
    #frame where its trajectory begins
    st=int(np.min(pc['frame']))
    #frame where its trajectory ends
    e=int(np.max(pc['frame']))
    #find an empty row (for the length of current track)
    er=np.where((np.sum(~np.isnan(allT[st-ofs:e-ofs+1,:,0]),axis=0))==0)[0][0]
    #er=np.where(np.isnan(allT[st,:,0]))[0][0]
    allT[range(st-ofs,e-ofs+1),er,0]=pc.x
    allT[range(st-ofs,e-ofs+1),er,1]=pc.y


#compare inter animal distances for larva from one frame vs. larva selected from random frames

#10x one frame

#IADfrList=np.round(range(FramesToProcess[0],FramesToProcess[-1],nFramesToProcess/10))
#IADAll=np.array([])
#for i in IADfrList:
#    t3x=np.array(t3['x'][t3['frame']==i])
#    t3y=np.array(t3['y'][t3['frame']==i])
#    xDist=distanceTimeSeries(t3x)
#    yDist=distanceTimeSeries(t3y)
#    IAD=np.sqrt(xDist**2+yDist**2)
#    IADAll=np.append(IADAll,IAD)
#
##Same number of animals but from random frames
#
#numAnimals=50
#
#IADAllr=np.array([])
#for i in range(10):
#    randPicks=list(np.round(np.random.uniform(0,np.shape(t3)[0],numAnimals)))
#    t3Rand=t3.iloc[randPicks,:]
#    t3xr=np.array(t3Rand['x'])
#    t3yr=np.array(t3Rand['y'])
#    xDistr=distanceTimeSeries(t3xr)
#    yDistr=distanceTimeSeries(t3yr)
#    IADr=np.sqrt(xDistr**2+yDistr**2)
#    IADAllr=np.append(IADAllr,IADr)
#plt.figure()
#histRaw=plt.hist(IADAll,100)
#histRand=plt.hist(IADAllr,100)
#plt.figure()
#plt.plot(histRaw[1][0:-1]/5,histRaw[0]/np.shape(IADAll),'k')
#plt.plot(histRand[1][0:-1]/5,histRand[0]/np.shape(IADAllr),'r')
#plt.title('Inter Animal Distance Distribution')
#plt.xlabel('IAD [mm]')
#plt.ylabel('fraction')

allT_d=np.diff(allT,axis=0)
travel=np.sqrt(allT_d[:,:,0]**2 + allT_d[:,:,1]**2)
travelb=travel.copy()
#travelb[np.where(travel < 1)]=0
travelb[np.where(travel >20)]=np.nan
plt.plot(np.nanmean(travelb[:-1],axis=1)) 

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    
    
c=moving_average(np.nanmean(travelb,axis=1),60)
plt.plot(c[:-1])
with open(pickleFile, 'w') as f:
    pickle.dump([c], f)

#aa=np.matlib.repmat(range(1,numAnimals),numAnimals-1,1)
#ba=np.matlib.repmat(range(numAnimals-1),numAnimals-1,1)
#bb=ba.T
#z=np.tril(np.ones(numAnimals-1,dtype='int'))
#
#NN=np.zeros(numAnimals)
#NNr=np.zeros(numAnimals)
#for i in range(1,numAnimals):
#    y=(np.triu(aa)==i)|(np.triu(bb)==i)
#    yb=y.T
#    zz=z|yb
#    nnList=np.flatnonzero(yb[np.where(zz)])
#    rList=list(np.round(np.random.uniform(1,5777,numAnimals)))
#    NN[i]=np.min(IAD[nnList])
#    NNr[i]=np.min(IAD[rList])
#tp.annotate(all_results, gray)
#plt.plot()
#tp.plot_traj(all_results)

#with tp.PandasHDFStore('data.h5') as s:
#    frame_2_results = s.get(2)

#frame_2_results.head()  # Display the first few rows.

#allAllX=[]
#allAllY=[]
#    
#for i in range(nFramesToProcess):
#    allX=[]
#    allY=[]
#    allX.append([x.pt[0] for x in keyAll[i]])
#    allY.append([x.pt[1] for x in keyAll[i]])
#    allAllX.append(allX)
#    allAllY.append(allY)
#    
#tAll=np.zeros(nFramesToProcess,150,2)
#ts=np.squeeze(np.asarray([allAllX[0],allAllY[0]]))
#tAll[0,0:np.shape(ts)[1],:]=ts
#
#def closest_node(node, nodes):
#    dist_2 = np.sum((nodes - node)**2, axis=1)
#    return np.argmin(dist_2)
#    
#for i in range(nFramesToProcess-1):
#    tnOld=np.squeeze(np.asarray([allAllX[i+1],allAllY[i+1]]))
#    tc=tAll[i,:,np.shape(tnOld)[1]]
#    tn=np.squeeze(np.asarray([allAllX[i+1],allAllY[i+1]]))
#    for j in range(np.shape(tc[1])):
#        a=tc[:,j]
#        cn=closest_node(a,tn.T)
#        tAll[i+1,j,:]=tn[:,cn]

with open('d:\speed_a.pickle', 'r') as f:
    ca=pickle.load(f)[2]

with open('d:\speed_b.pickle', 'r') as f:
    cb=pickle.load(f)[2]
    
with open('d:\speed_c.pickle', 'r') as f:
    cd=pickle.load(f)[2]
    
with open('d:\speed_d.pickle', 'r') as f:
    ce=pickle.load(f)[2]