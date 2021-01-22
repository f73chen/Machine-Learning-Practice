# R = min(L1, L2), where L are the eigenvalues
# if greater than threshold Lmin, considered a corner
# cv2.goodFeaturesToTrack() finds N strongest corners in the image
    # image
    # number of corners to find
    # quality level (0-1) --> min quality of acceptable corner
    # min euclidean distance between corners
# first, finds all corners in image
    # reject all below min quality
    # sorts remaining based in descending quality
    # takes first strongest corner, removes all nearby corners in ran of min distance
    # returns the N strongest corners

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('chess.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 5, 255, -1)

plt.imshow(img), plt.show()