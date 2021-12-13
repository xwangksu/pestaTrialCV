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
# Detect each individual image
for imf in rgbIm:        
    imgFile = cv2.imread(workingPath+imf)

    croppedImage = imgFile[(int(imgFile.shape[0]*0.2)):(int(imgFile.shape[0]*0.8)-1), 0:(int(imgFile.shape[1]-1)), ]        
    
    print(targetPath+imf)
    cv2.imwrite(targetPath+imf, croppedImage)