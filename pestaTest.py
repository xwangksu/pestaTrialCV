import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

filename = '/home/xuwang1/Documents/Pesta_trials/pct_crop_HSVForeGround/GOPR1565.JPG'
# Loads an image
raw = cv2.imread(cv2.samples.findFile(filename))

# # Smooth (blur) images
# blurred = cv2.bilateralFilter(raw,7,80,80)

# cv2.imshow('Blurred', blurred)
# cv2.waitKey(0)

kernel1 = np.ones((3,3),np.uint8)
kernel2 = np.ones((7,7),np.uint8)
kernel3 = np.ones((9,9),np.uint8)
kernel4 = np.ones((11,11),np.uint8)

# dilation = cv2.dilate(raw, kernel1, iterations = 1)
# cv2.imshow('Dilation', dilation)
# cv2.waitKey(0)

# opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel3)
# cv2.imshow('Opened', opened)
# cv2.waitKey(0)

closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel3)
# cv2.imshow('Closed', closed)
# cv2.waitKey(0)

opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel4)
cv2.imshow('Opened', opened)
cv2.waitKey(0)

# erosion = cv2.erode(raw, kernel1, iterations = 1)
# cv2.imshow('Erosion', erosion)
# cv2.waitKey(0)

