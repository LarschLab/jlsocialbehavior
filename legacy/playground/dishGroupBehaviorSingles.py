__author__ = 'jlarsch'

#import Tkinter
import tkFileDialog
import sys
sys.path.append('C:/Users/johannes/Dropbox/python/zFishBehavior/dishGroupBehavior/')
import joFishHelper
import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

batchmode=1
timeStim=1

if batchmode:
    ExpFile=tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b'))
    PdfFile=ExpFile[:-4]+'.pdf'
    df=pd.read_csv(ExpFile,sep=',')
    experiment=[];
    
    
        
    for index, row in df.iterrows():
        print 'processing: ', row['aviPath']
        currAvi=row['aviPath']
        experiment.append(joFishHelper.experiment(currAvi))
        #experiment[index].plotOverview()
        #pdf.savefig()  # saves the current figure into a pdf page
        #plt.close()
    
    #pair up each single animal with another single animal from another experiment
    
    loopList=np.roll(range(14),1)
    expDur=30*60*60
    for index, row in df.iterrows():
        pairedTrajectories=np.append(experiment[index].rawTra[range(expDur),:,:],experiment[loopList[index]].rawTra[range(expDur),:,:],axis=1)
        experiment[index].Pair=joFishHelper.Pair(pairedTrajectories,experiment[index].expInfo)
        experiment[index].sPair=joFishHelper.shiftedPair(experiment[index].Pair,experiment[index].expInfo)
        experiment[index].ShoalIndex=(experiment[index].sPair.spIAD_m-experiment[index].Pair.IAD_m)/experiment[index].sPair.spIAD_m
        experiment[index].totalPairTravel=sum(experiment[index].Pair.totalTravel)
    
    with PdfPages(PdfFile) as pdf:
        for index, row in df.iterrows():
            print 'processing: ', row['aviPath']
            experiment[index].plotOverview()
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()        
        
        df['ShoalIndex']=pd.DataFrame([f.ShoalIndex for f in experiment])
        df['totalTravel']=pd.DataFrame([f.totalPairTravel for f in experiment])
        IADall=[]
        IADall.append([x.Pair.IAD[0:30*60*60] for x in experiment])
        t=np.nanmean(IADall,axis=1)
        smIAD_mAll=np.nanmean([f.sPair.spIAD_m for f in experiment])
        
        fig, axes = plt.subplots(nrows=1, ncols=2)
        
        sns.boxplot(ax=axes[0],x='condition',y='ShoalIndex',data=df,width=0.2)
        sns.stripplot(ax=axes[0],x='condition',y='ShoalIndex',data=df, size=4,jitter=True,edgecolor='gray')

        sns.boxplot(ax=axes[1],x='condition',y='totalTravel',data=df,width=0.2)
        sns.stripplot(ax=axes[1],x='condition',y='totalTravel',data=df, size=4,jitter=True,edgecolor='gray')
        
        axes[0].set_ylim(-0.1,1)
        axes[1].set_ylim(0,)
        pdf.savefig()
        
        if timeStim:
            fig=plt.figure(figsize=(8, 2))
            x=np.arange(float(np.shape(t[0])[0]))/(experiment[0].expInfo.fps*60)
            stim=np.arange(float(np.shape(t[0])[0]))/(experiment[0].expInfo.fps*60)
            plt.plot(x,t[0])
            plt.plot([0, np.shape(t[0])[0]], [smIAD_mAll, smIAD_mAll], 'r:', lw=1)
            plt.xlim([0, 100])
            plt.ylim([0, 100])
            plt.xlabel('time [minutes]')
            plt.ylabel('IAD [mm]')
            plt.title('Inter Animal Distance over time')
        
        pdf.savefig()
else:    
    avi_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b'))
    experiment=joFishHelper.experiment(avi_path)
    experiment.plotOverview()