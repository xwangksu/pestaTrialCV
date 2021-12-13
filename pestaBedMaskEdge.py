import cv2
import argparse
import os
# import matplotlib.pyplot as plt
import numpy as np
import csv
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

finalFile = open(str(targetPath)+'UpEdge.csv', 'wt')

# print(float(cp[1]))
imageFiles = os.listdir(workingPath)
rgbIm = []
for im in imageFiles:
    if im.find(".JPG") != -1:
        rgbIm.append(im)

try:
    writer = csv.writer(finalFile, delimiter=',', lineterminator='\n')
    writer.writerow(('Images','xu0','yu0','xu1','yu1'))
    # Detect each individual image
    for imf in rgbIm:        
        imgFile = cv2.imread(workingPath+imf)

        xc = int(imgFile.shape[1]*0.5)
        yc = int(imgFile.shape[0]*0.5)
        dist0 = imgFile.shape[0]

        # Edge detection
        dst = cv2.Canny(imgFile, 100, 200, apertureSize = 3)

        # Copy edges to the images that will display the results in BGR
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)

        # Probabilistic Line Transform
        linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, threshold = 100, lines = None, minLineLength = 200, maxLineGap = 70)

        print(targetPath+imf)
        
        # Draw the lines
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                # cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
                # print(l[0], ' ', l[1], ' ', l[2], ' ', l[3])
                if (l[1] <= yc) or (l[3] <= yc):
                    p1 = np.array([l[0], l[1]])
                    p2 = np.array([l[2], l[3]])
                    p3 = np.array([xc, yc])
                    dist_c = np.abs(np.cross(p2 - p1, p3 - p1))/np.linalg.norm(p2 - p1)
                    if dist0 >= dist_c:
                        dist0 = dist_c
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                if (l[1] <= yc) or (l[3]<=yc):
                    p1 = np.array([l[0], l[1]])
                    p2 = np.array([l[2], l[3]])
                    p3 = np.array([xc, yc])
                    dist_c = np.abs(np.cross(p2 - p1, p3 - p1))/np.linalg.norm(p2 - p1)
                    if dist0 == dist_c:
                        k = (l[3] - l[1])/(l[2] - l[0])
                        b = l[1] - l[0] * k
                        ys = int(k*1 + b)
                        ye = int(k*(imgFile.shape[1]-1) + b)
                        print(ys, ' ', ye)
                        cv2.line(imgFile, (1, ys), (imgFile.shape[1]-1, ye), (255,0,0), 3, cv2.LINE_AA)
                        writer.writerow((imf, 1, ys, imgFile.shape[1]-1, ye))
                        break

        while True:
            # cv2.imshow("Source", src)
            cv2.imshow("Detected Lines (in blue) - Probabilistic Line Transform", imgFile)
            if cv2.waitKey(1) == ord('q'):
                break
finally:
    finalFile.close()        
    # cv2.imwrite(targetPath+imf, opened)

