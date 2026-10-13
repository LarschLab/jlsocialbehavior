# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:33:38 2016

@author: jlarsch
"""

import cv2
image_path= 'd:/DSC_5839.JPG'
img = cv2.imread(image_path,cv2.IMREAD_COLOR)

#display image before thresholding
cv2.imshow('I am an image display window',img)
cv2.waitKey(0)

#convert image to gray scale - needed for thresholding
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#apply threshold to gray image to obtain binary image

threshold=150 #value above which pixel values will be set to max_value
max_value=255  #value to which pixels above threshold will be set
threshold_stype=cv2.THRESH_BINARY #default threshold method

ret, img_binary = cv2.threshold(img_gray, threshold, max_value, threshold_stype)

#display image after thresholding
cv2.imshow('image after applying threshold',img_binary)
cv2.waitKey(0)

#save the binary image
cv2.imwrite('d:/binary.png',img_binary)
cv2.destroyAllWindows()

img1 = img[47:151, 106:157,:]
img2 = img[148:252, 207:258,:]
diff1 = cv2.absdiff(img1, img2)
plt.imshow(diff1)