# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:08:53 2017

@author: jlarsch
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IADall=[]
IADall.append([x.pair.IAD()[0:30*60*50] for x in expSet.experiments])
t=np.nanmean(IADall,axis=1)
smIAD_mAll=np.nanmean([f.spIAD_m() for f in expSet.experiments])
x=np.arange(float(np.shape(t[0])[0]))/(expSet.experiments[0].expInfo.fps*60)
#stim=np.arange(float(np.shape(t[0])[0]))/(experiment[0].expInfo.fps*60)
plt.plot(x,t[0])
plt.plot([0, np.shape(t[0])[0]], [smIAD_mAll, smIAD_mAll], 'r:', lw=1)
plt.xlim([0, 100])
plt.ylim([0, 100])
plt.xlabel('time [minutes]')
plt.ylabel('IAD [mm]')
plt.title('Inter Animal Distance over time')

rng = pd.date_range('1/10/2011', periods=len(t[0,:]), freq='33ms')
ts=pd.Series(t[0,:],index=rng)
ts.rolling('1min').mean().plot()
ts.resample('10min').plot('bar')

si=np.array([x.ShoalIndex() for x in expSet.ee])
si_m=si.reshape((-1,5)).mean(axis=0)
si_std=si.reshape((-1,5)).std(axis=0)

x=np.arange(5)+1
plt.bar(x, si_m, yerr=si_std, width=0.5,color='k')

lims = plt.ylim()
plt.ylim([0, lims[1]]) 
plt.xlim([0.5, 6]) 
plt.ylabel('shoaling index +/- SD')
plt.xticks(x+.25,['C-skype','O-replay_r','O-replay_c','O-circle_c','O-circle_r'])
plt.title('Attraction to black disk, closed loop skype vs. open loop \n open loop: real vs. constant speed')