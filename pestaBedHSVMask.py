
import cv2
import argparse
import os
# import matplotlib.pyplot as plt
import numpy as np
# from scipy.signal import find_peaks

#------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--srcPath", required=True,
    help="source image folder")
ap.add_argument("-t", "--tgtPath", required=True,
    help="target folder to save the maker list")

args = ap.parse_args()
workingPath = args.srcPath
targetPath = args.tgtPath

# print(float(cp[1]))
imageFiles = os.listdir(workingPath)
rgbIm = []
for im in imageFiles:
    if im.find(".JPG") != -1:
        rgbIm.append(im)

scale_percent = 30 # percent of original size


# Detect each individual image
for imf in rgbIm:        
    imgFile = cv2.imread(workingPath+imf)
  
    # resize image
    width = int(imgFile.shape[1] * scale_percent / 100)
    height = int(imgFile.shape[0] * scale_percent / 100)
    dim = (width, height)
    resizedImg = cv2.resize(imgFile, dim, interpolation = cv2.INTER_AREA)

    # HSV mask generation
    imgHSV = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2HSV)

    l_b = np.array([0, 12, 140])
    u_b = np.array([14, 38, 255])

    fgMask = cv2.inRange(imgHSV, l_b, u_b)

    print(targetPath+imf)
    cv2.imwrite(targetPath+imf, fgMask)

