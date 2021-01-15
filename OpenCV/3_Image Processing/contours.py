# contours are curves joining all continous points along the boundary,
    # having the same colour or intensity
# useful in shape analysis and object detection and recognition
# first apply threshold or canny edge detection to convert image to binary
# findContours modifies the source image, so store source to some other variable
# object to be found should be white, and background should be black

import numpy as np
import cv2

img = cv2.imread('original.png')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)

# arg1: source image
# arg2: contour retrieval mode
# arg3: contour approximation method
# output contours is a python list of all contours in the image
    # each is a numpy array (x, y) of boundary poitns of the object
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# drawContours can be used to draw any shape given boundary points
# arg1: source image
# arg2: contours passed as a python list
# arg3: index of contours (useful for drawing individual contours)
    # to draw all, pass -1
# arg4: colour
# arg5: thickness
#img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)   # draw all contours

img = cv2.drawContours(img, contours, 3, (0, 255, 0), 3)    # draw only the 4th contour

#cnt = contours[4]
#img = cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)   # also draw only the 4th contour

# how much of the coordinates to store is given by the contour approximation method
    # cv2.CHAIN_APPROX_NONE --> all boundary points are stored
    # cv2.CHAIN_APPROX_SIMPLE --> removes all redundant points (ex. 3 collinear points) and compresses contour

cv2.imshow('contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()