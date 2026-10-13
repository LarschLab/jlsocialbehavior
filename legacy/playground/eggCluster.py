# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:30:20 2016

@author: jlarsch
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import os

start_from_scratch=0
store_individual_eggs=0
get_categories=1

def tileImage(data,numImages):
    numRC=np.ceil(np.sqrt(numImages))
    largestSize=np.max([img.shape for img in data[:numImages]])
    output=np.zeros((numRC*largestSize,numRC*largestSize))
    for i,im in enumerate(data[:numImages]):
        
        startx=largestSize*np.mod(i,numRC)
        starty=np.floor(i/numRC)*largestSize
        print i,startx,starty
        output[startx:startx+im.shape[0],starty:starty+im.shape[1]]=im[:,:]
    return output
    


if start_from_scratch:
    imlist=enumerate(glob.iglob('d:/data/eggs/clean/*.tif'))
    
    allEggs=[]
    
    for (i,image_file) in imlist:
        print 'reading from ',image_file
        image = cv2.imread(image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 20,minRadius =50,maxRadius=100)
        circles = np.round(circles[0, :]).astype("int")
        
        for (x, y, r) in circles:
            allEggs.append(image[y-r:y+r,x-r:x+r])
else:
    imlist=enumerate(glob.iglob('d:/data/eggs/single/*.jpg'))
    allEggs=[]
    
    for (i,image_file) in imlist:
        print 'reading image '+str(i)
        image = cv2.imread(image_file)
#        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image is not None:
            allEggs.append(image[1])
        else:
            allEggs.append(np.zeros((1,1)))

if store_individual_eggs:
    #store all images

    for i,egg in enumerate(allEggs):
        fileName='d:/data/eggs/single/egg_'+str(i).zfill(3)+'.jpg'
        cv2.imwrite(fileName,egg)
    
    
eggTile=tileImage(allEggs,len(allEggs))
plt.imshow(eggTile,cmap='gray')




if get_categories:
    
    categoryFile='d:/data/eggs/single/categorySave.csv'
    
    if np.equal(~os.path.isfile(categoryFile),-2):
        category=pd.read_csv(categoryFile,index_col=0)
        nextImage=len(category.index)
    else:
        
        category=pd.DataFrame({'imageNr': 0, 'path':0,'category':0},index=range(0))
        nextImage=0
        
    
    plt.ion()
    imlist=list(glob.iglob('d:/data/eggs/single/*.jpg'))
    numImages=len(imlist)
    
    while nextImage < numImages:
#    while nextImage < 5:
        image = cv2.imread(imlist[nextImage])
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            plt.imshow(gray,cmap='gray')
            plt.pause(0.05)
            
            usrInput=(raw_input('category '+'for image '+str(nextImage).zfill(3)+' of ' + str(numImages)+' : '))
        else:
            usrInput='n'
        newCat=pd.DataFrame({'imageNr': nextImage, 'path':imlist[nextImage],'category':usrInput},index=range(1))
        category=category.append(newCat,ignore_index=True)
        if np.mod(nextImage,10)==0:        
            category.to_csv(categoryFile)
        nextImage +=1
        
#
#image = cv2.imread('d:/data/egg_eg.tif')
#plt.imshow(image)
#image = cv2.imread('d:/data/egg_eg.tif')
#output = image.copy()
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
## detect circles in the image
#circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 20,minRadius =50,maxRadius=100)
#cv2.namedWindow("output", cv2.WINDOW_NORMAL)
#
## ensure at least some circles were found
#if circles is not None:
#	# convert the (x, y) coordinates and radius of the circles to integers
#	circles = np.round(circles[0, :]).astype("int")
# 
#	# loop over the (x, y) coordinates and radius of the circles
#	for (x, y, r) in circles:
#		# draw the circle in the output image, then draw a rectangle
#		# corresponding to the center of the circle
#		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
#		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
#     
#	cv2.imshow("output", np.hstack([image, output]))
#	cv2.waitKey(0)