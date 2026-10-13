# -*- coding: utf-8 -*-
"""
Created on Tue May 03 09:24:30 2016

@author: jlarsch
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 4


img1 = cv2.imread('d:/one_animal_rotated.jpg',0)          # queryImage
img2 = cv2.imread('d:/many_animals.jpg',0) # trainImage


# Initiate SIFT detector
#sift = cv2.xfeatures2d.SIFT_create(0,3,0,100,2)
sift = cv2.xfeatures2d.SIFT_create(0,3,0)
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

img_kp=np.zeros(np.shape(img2),dtype='uint8')
img_kp=cv2.drawKeypoints(img2,kp2,img_kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#plt.imshow(img_kp)

print("# kp1: {}, descriptors: {}".format(len(kp1), des1.shape))
print("# kp2: {}, descriptors: {}".format(len(kp2), des2.shape))


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
        

#src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
#dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
#
#M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#matchesMask = mask.ravel().tolist()
#
#
#
#h,w = img1.shape
#pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#dst = cv2.perspectiveTransform(pts,M)
#
#img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.figure()
plt.imshow(img3, 'gray'),plt.show()