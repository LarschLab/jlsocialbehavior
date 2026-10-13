# -*- coding: utf-8 -*-

"""

Created on Thu Dec 18 20:42:12 2014



@author: dalmaschio

"""



import numpy as np

#matplotlib.use('GTK')

import matplotlib



import matplotlib.pyplot as plt



import scipy as sc

import matplotlib.cm as cmap

from skimage import io

from skimage.viewer import viewers

import skimage as sk

import matplotlib.cm as cmap

from tifffile import imsave

import pyqtgraph as pg



#img = io.MultiImage('F9_pmtUG-2.tif')

#img = io.MultiImage('is2b_G6s_6dpf_001_8bit.tif')

img = io.MultiImage('M10_ms.tif')

frames=sk.img_as_ubyte(img)

print 'Data loaded'

Height= len(frames[0,:,0])

Width= len(frames[0,0,:])

Samples= len(frames[:,0,0])

print   'Planes,Height,Width'

print   Samples,Height,Width



if Samples==1:

    #Samples=Width

    series=frames[0,:,:]

    DFFseries=np.empty((Height,Width))

    

    for i in range(Height):

      z=np.std(series[i,:])

      if z>10:

          DFFseries[i,:]=(series[i,:]-np.mean(series[i,0:15]))/np.mean(series[i,0:15])#region where to define the baseline

      else :

          DFFseries[i,:]=(series[i,:]-np.mean(series[i,0:15]))/np.mean(series[i,0:15])#region where to define the baseline



    imsave('DFFlinescan.tif', DFFseries)

    #plt.figure(),plt.plot(base),plt.plot(DFFseries[180]),plt.imshow(DFFseries),plt.colorbar(),plt.show()# to be adapt in vcase of background

    print 'DFF linescan calculated and saved'

else :

    series=np.empty((Width*Height,len(frames[:,0,0])))

    series=np.reshape(frames,(len(frames[:,0,0]),Width*Height))

    series=np.transpose(series)

    DFFseries=np.empty((Height*Width,Samples))

    for i in range(Height*Width):

      z=np.std(series[i,:])

      if z>10:

          #DFFseries[i,:]=(series[i,:]-(np.mean(series[i,:])-0.5*np.std(series[i,:])))/(np.mean(series[i,:])-0.5*np.std(series[i,:]))

          DFFseries[i,:]=(series[i,:]-np.mean(series[i,0:15]))/np.mean(series[i,0:15])#region where to define the baseline

      else :

          DFFseries[i,:]=0 

          DFFseries[i,:]=(series[i,:]-np.mean(series[i,0:15]))/np.mean(series[i,0:15])#region where to define the baseline

    print 'DFF 2Dscan calculated and saved'

    DFFimage=np.zeros((len(frames[:,0,0]),Height,Width,),dtype=np.float)

    DFFimage= np.reshape(np.transpose(DFFseries),(len(frames[:,0,0]),Height,Width))

    imsave('DFF_2Dscan.tif', DFFimage)

    

box_pts=5#window size for smoothing

box = np.ones(box_pts)/box_pts



t=183#sampling time in msec

time=np.multiply(183*np.ones((Width)),range(Width))/1000# vector with the sample times



# in case you have linescan acquisition use DFFseries adressed only setting in the first place the y range measured in fiji

if Samples==1:

    r=np.average(DFFseries[1800:1960,:],axis=(0))# axis are inverted with respect to fiji

else:

    print 'free'

# in case you have 2D acquisition use DFFimage adressed setting in the second place the y range and in the third the x range measured in fiji

    r=np.average(DFFimage[:,2:5,2:5],axis=(1,2))# axis are inverted with respect to fiji first axis is time, than come y from fiji and last x from fiji

filt= np.convolve(r, box, mode='same')

plt.figure(), plt.plot(time,r,color='r'),

plt.plot(time,filt,color='b'),#comment this line if don't want to display 

plt.savefig('myfig.eps')

plt.show()



imv = pg.ImageView()

imv.setImage(frames)

imv.show()