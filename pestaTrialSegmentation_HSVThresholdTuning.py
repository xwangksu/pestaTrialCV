import cv2
import math
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('Trackbars')
cv2.moveWindow('Trackbars', 2020, 0)
cv2.createTrackbar('hueLower', 'Trackbars', 50, 179, nothing)
cv2.createTrackbar('hueHigher', 'Trackbars', 100, 179, nothing)
cv2.createTrackbar('satLow', 'Trackbars', 100, 255, nothing)
cv2.createTrackbar('satHigh', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('valLow', 'Trackbars', 100, 255, nothing)
cv2.createTrackbar('valHigh', 'Trackbars', 100, 255, nothing)
# cv2.imshow('Trackbars')

filename = '/home/xuwang1/Documents/Pesta_trials/pft_crop/GOPR1643.JPG'
# Loads an image
raw = cv2.imread(cv2.samples.findFile(filename))
img = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_GRAYSCALE)

scale_percent = 25 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
raw_resized = cv2.resize(raw, dim, interpolation = cv2.INTER_AREA)
src = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# Check if image is loaded fine
if src is None:
    print('Error opening image!')
    print('Usage: hough_lines.py [image_name -- default ' + filename + '] \n')

while True:
    cv2.imshow("Raw_Image", raw_resized)
    hsv_raw_resized = cv2.cvtColor(raw_resized, cv2.COLOR_BGR2HSV)

    hueLow = cv2.getTrackbarPos('hueLower', 'Trackbars')
    hueUp = cv2.getTrackbarPos('hueHigher', 'Trackbars')
    sLow = cv2.getTrackbarPos('satLow', 'Trackbars')
    sHigh = cv2.getTrackbarPos('satHigh', 'Trackbars')
    vLow = cv2.getTrackbarPos('valLow', 'Trackbars')
    vHigh = cv2.getTrackbarPos('valHigh', 'Trackbars')

    l_b = np.array([hueLow, sLow, vLow])
    u_b = np.array([hueUp, sHigh, vHigh])

    fgMask = cv2.inRange(hsv_raw_resized, l_b, u_b)
    cv2.imshow('fgMask', fgMask)
    cv2.moveWindow('fgMask', 0, 700)

    foreGround = cv2.bitwise_and(raw_resized, raw_resized, mask = fgMask)
    # cv2.imshow('ForeGround', foreGround)
    # cv2.moveWindow('foreGround', 1000, 0)

    bgMask = cv2.bitwise_not(fgMask)
    cv2.imshow('bgMask', bgMask)
    cv2.moveWindow('bgMask', 1200, 700)

    backGround = cv2.cvtColor(bgMask, cv2.COLOR_GRAY2BGR)

    final_raw_img = cv2.add(foreGround, backGround)
    cv2.imshow('Final', final_raw_img)
    cv2.moveWindow('Final', 1200, 0)


    # cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    if cv2.waitKey(1) == ord('q'):
        break

    
    # imgHSV = cv2.cvtColor(imgFile, cv2.COLOR_BGR2HSV)

    # l_b = np.array([26, 21, 9])
    # u_b = np.array([78, 246, 254])

    # fgMask = cv2.inRange(imgHSV, l_b, u_b)

    # indices = np.where(fgMask!= [0])
    # y_coordinates = indices[0]
    # # print(y_coordinates)
    # hist_y = np.histogram(indices[0], bins = 200)

    # plt.hist(y_coordinates, density=True, bins=100)  # density=False would make counts
    
    # plt.ylabel('Probability')
    # plt.xlabel('Y_Coord')
    # print(hist_y[0])
    # peaks, _ = find_peaks(hist_y[0], height = 5000, distance = 40)
    # plt.plot(peaks, hist_y[0][peaks], "x")
    # print(peaks)
    # print(np.mean(peaks))

    # plt.show()
    # cv2.imshow('fgMask', fgMask)
    # cv2.moveWindow('fgMask', 0, 1000)
    # while True:
    #     if cv2.waitKey(1) == ord('q'):
    #         break



