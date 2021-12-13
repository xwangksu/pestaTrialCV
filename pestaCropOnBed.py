import cv2
import argparse
import os
# import matplotlib.pyplot as plt
import numpy as np
import csv
from numpy.lib.polynomial import poly
import pandas as pd
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

# finalFile = open(str(targetPath)+'UpEdge.csv', 'wt')

dfUpEdges= pd.read_csv(str(workingPath)+'UpEdge.csv', usecols=['Images', 'yu0', 'yu1'])
dfDownEdges = pd.read_csv(str(workingPath)+'DownEdge.csv', usecols=['Images', 'yd0', 'yd1'])
# print(dfUpEdges)
dfBothEdges = dfUpEdges.merge(dfDownEdges, left_on='Images', right_on='Images', how='left')

print(dfBothEdges)

# # print(float(cp[1]))
imageFiles = os.listdir(workingPath)
rgbIm = []
for im in imageFiles:
    if im.find(".JPG") != -1:
        rgbIm.append(im)

# # try:
# #     writer = csv.writer(finalFile, delimiter=',', lineterminator='\n')

# Detect each individual image
for imf in rgbIm:        
    imgFile = cv2.imread(workingPath+imf)
    imgHeight = imgFile.shape[0]
    imgWidth = imgFile.shape[1]

    # print(dfBothEdges['Images'].str)
    if dfBothEdges['Images'].str.contains(imf).any():
        # print('yes')
        yu0 = int(dfBothEdges['yu0'][dfBothEdges['Images'] == imf])
        yu1 = int(dfBothEdges['yu1'][dfBothEdges['Images'] == imf])
        yd0 = int(dfBothEdges['yd0'][dfBothEdges['Images'] == imf])
        yd1 = int(dfBothEdges['yd1'][dfBothEdges['Images'] == imf])
    else:
        yu0 = 0
        yu1 = 0
        yd0 = imgHeight
        yd1 = imgHeight
    print(imf, ' ', yu0, ' ', yu1, ' ', yd0, ' ', yd1)
    mask = np.zeros((imgHeight, imgWidth), dtype=np.uint8)
    polyPoints = np.array([[[0, yu0], [imgWidth, yu1], [imgWidth, yd1], [0, yd0]]])
    cv2.fillPoly(mask, polyPoints, (255))

    cropped = cv2.bitwise_and(imgFile, imgFile, mask = mask)
    # while True:
    #     # cv2.imshow("Source", src)
    #     cv2.imshow("Croppred", cropped)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    cv2.imwrite(targetPath+imf, cropped)




