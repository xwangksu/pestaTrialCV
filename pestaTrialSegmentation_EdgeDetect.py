import cv2
import math
import numpy as np

filename = '/home/xuwang1/Documents/Pesta_trials/pct_crop_DsEdgeMaskFill/GOPR1508.JPG'
# Loads an image
img = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_GRAYSCALE)

scale_percent = 100 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
src = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# Check if image is loaded fine
if src is None:
    print('Error opening image!')
    print('Usage: hough_lines.py [image_name -- default ' + filename + '] \n')

xc = int(src.shape[1]*0.5)
yc = int(src.shape[0]*0.5)

# Edge detection
dst = cv2.Canny(src, 100, 200, apertureSize = 3)

# Copy edges to the images that will display the results in BGR
cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

# Probabilistic Line Transform
linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, threshold = 100, lines = None, minLineLength = 200, maxLineGap = 70)

# Draw the lines
dist0 = src.shape[0]
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        # cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
        # print(l[0], ' ', l[1], ' ', l[2], ' ', l[3])
        if (l[1] >= yc) or (l[3]>=yc):
            p1 = np.array([l[0], l[1]])
            p2 = np.array([l[2], l[3]])
            p3 = np.array([xc, yc])
            dist_c = np.abs(np.cross(p2 - p1, p3 - p1))/np.linalg.norm(p2 - p1)
            if dist0 >= dist_c:
                dist0 = dist_c
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        if (l[1] >= yc) or (l[3]>=yc):
            p1 = np.array([l[0], l[1]])
            p2 = np.array([l[2], l[3]])
            p3 = np.array([xc, yc])
            dist_c = np.abs(np.cross(p2 - p1, p3 - p1))/np.linalg.norm(p2 - p1)
            if dist0 == dist_c:
                k = (l[3] - l[1])/(l[2] - l[0])
                b = l[1] - l[0] * k
                ys = int(k*1 + b)
                ye = int(k*(src.shape[1]-1) + b)
                if ye > src.shape[0]:
                    ye = src.shape[0]
                print(ys, ' ', ye)
                cv2.line(cdstP, (1, ys), (src.shape[1]-1, ye), (255,0,0), 3, cv2.LINE_AA)
                break

while True:
    # cv2.imshow("Source", src)
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    if cv2.waitKey(1) == ord('q'):
        break



