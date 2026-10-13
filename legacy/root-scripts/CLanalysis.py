# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 22:51:02 2017

@author: jlarsch
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as sta


#csvFile=tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b'))
currTxt=csvFile
rawData=pd.read_csv(currTxt,header=None,delim_whitespace=True)

# with open(csvFile) as f:
#                print(self.expInfo.trajectoryPath)
#                mat=np.loadtxt((x.replace(b'   ',b',').replace(b' ',b'') for x in f if len(x)>6),delimiter=',')

df=pd.DataFrame(mat,columns=['x','y','o','xp','yp','s'])
df['episode']=np.repeat(np.arange(mat.shape[0]/67.),67)

fig, ax = plt.subplots()
t=df[df.episode==20]
t.plot(x='x',y='y',lineStyle=':',ax=ax,marker='o')
t.plot(x='xp',y='yp',lineStyle=':',ax=ax,color='r',marker='o')
plt.plot(t.x.values[0],t.y.values[0],'ok')
plt.plot(t.xp.values[0],t.yp.values[0],'ok')
ax.set_aspect('equal')
ax.set_xlim(t[['x','xp']].values.min(),t[['x','xp']].values.max())

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho
def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

fig, ax = plt.subplots()
co=sns.color_palette("hls", 34)
#for i in range(int(mat.shape[0]/67)):
traAll=np.array([1,1,1,1,1])
for i in range(500):
    cdf=df[df.episode==i]
    s=cdf.s.values
    col=co[int(s[0])]
    mid=[cdf.x.values[0],cdf.y.values[0]]
    x=cdf[['x','xp']]-mid[0]
    y=cdf[['y','yp']]-mid[1]
    traX=x.values
    traY=y.values
    traT,traR=cart2pol(traX,traY)
    traT=traT-cdf['o'].values[0]
    x,y=pol2cart(traT,traR)
    traAll=np.vstack((traAll,np.array([x[:,0],y[:,0],x[:,1],y[:,1],s]).T))
    
traAll=traAll[1:,:]
traAll

plt.scatter(x=traAll[:,0],y=traAll[:,1],c=traAll[:,4],s=7,alpha=0.5,cmap='viridis')
plt.colorbar()

plt.plot(traAll[-30:,2],traAll[-30:,3],'or',alpha=0.8,markerSize=4)
ax.set_aspect('equal')


dxy=traAll[:,[0,1]]-traAll[:,[2,3]]
d=np.sqrt(np.power(dxy[:,0],2)+np.power(dxy[:,1],2))
sall=np.array([df[df.episode==i].s.values[0] for i in range(500)])
dm=np.reshape(d,(-1,67))
sbins=np.arange(0,31,1)


co=sns.color_palette("viridis", sbins.shape[0])
cm=[]
cmed=[]
l=[]
plt.figure(2)
plt.clf()
for i in range(sbins.shape[0]-1):
    idx=((sall>sbins[i]) * (sall<=sbins[i+1]))
    cmed.append(np.median(dm[idx,:],axis=0))
    cm.append(np.mean(dm[idx,:],axis=0))
    plt.plot(cm[i],label=str(sbins[i+1]),color=co[i])


xr=range(10)
yr=np.nanmean(np.array(cm)[:,xr],axis=0)

slope, intercept, r_value, p_value, std_err = sta.linregress(xr,yr)
xNew=np.arange(67)
yNew=slope*xNew+intercept
plt.plot(xNew,yNew,label='fit_start',color='r')
plt.legend()

plt.figure(4)
plt.clf()
for i in range(sbins.shape[0]-1):
    plt.plot(cm[i]-yNew,color=co[i],label=str(sbins[i]))    
    
plt.legend()


yr=np.nanmedian(np.array(cmed)[:,xr],axis=0)

slope, intercept, r_value, p_value, std_err = sta.linregress(xr,yr)
xNew=np.arange(67)
yNewMed=slope*xNew+intercept


plt.figure(5)
plt.clf()
for i in range(sbins.shape[0]-1):
    plt.plot(cmed[i]-yNewMed,color=co[i],label=str(sbins[i]))    
    


plt.figure(7)
plt.clf()
for i in range(dm.shape[0]):
    
    plt.plot(dm[i,:]-yNewMed,color=co[int(np.round(sall[i]))],alpha=0.5)
    

plt.figure(8)
plt.clf()
sns.boxplot(x=np.round(sall).astype('int'),y=dm[:,50]-yNewMed[50])
sns.swarmplot(x=np.round(sall).astype('int'),y=dm[:,50]-yNewMed[50],linewidth=2,edgecolor='gray')
plt.xlim([-1,30])
plt.ylim([-20,20])

#fraction below zero = attraction
sallR=np.round(sall).astype('int')
y=dm[:,50]-yNewMed[50]
sizeCount=np.array([np.sum(sallR==i) for i in np.unique(sallR)])
n_attract=np.array([np.sum(y[sallR==i]<0) for i in np.unique(sallR)])
plt.figure(9)
plt.clf()
plt.plot(np.unique(sallR),n_attract/sizeCount.astype('float'),'o')
plt.ylim([0,1])
plt.xlim([0,32])

plt.figure(10)
plt.clf()
plt.plot(np.unique(sallR),(n_attract-(sizeCount-n_attract))/sizeCount.astype('float'),'o')
plt.ylim([-1,1])
plt.xlim([0,32])