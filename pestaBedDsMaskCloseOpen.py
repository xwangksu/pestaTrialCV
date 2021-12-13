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

kernel3 = np.ones((5,5),np.uint8)
kernel4 = np.ones((9,9),np.uint8)
kernel5 = np.ones((11,11),np.uint8)


# Detect each individual image
for imf in rgbIm:        
    imgFile = cv2.imread(workingPath+imf)

    closed = cv2.morphologyEx(imgFile, cv2.MORPH_CLOSE, kernel5)

    # opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel4)

    print(targetPath+imf)
    cv2.imwrite(targetPath+imf, closed)

