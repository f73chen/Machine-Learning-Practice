# 1. noise reduction
# edge detection is susceptible to noise in image
# use 5x5 gaussian filter

# 2. finding intensity gradient of the image
# filtered wit Sobel kernel in both x and y directions to get first derivatives
    # edge_gradient(G) = sqrt(Gx^2 + Gy^2)
    # angle(theta) = tan^-1 (Gy/Gx)
# gradient direction always perpendicular to edges
# rounded to one of 4 angles rep. vertical, horizontal, and 2 diagonal directions

# 3. non-maximum suppression
# after getting gradient magnitide and direction,
# scan full image to prevent unwanted pixels which don't constitute the edge
# at every pixel, check if it's a local maximum in the direction of the gradient
# suppose points A, B, C in line, with A on the edge of shape
    # gradient is parallel to A, B, C, and normal to edge
    # point A is checked with points B and C to see if it's a local maximum
    # if yes, remains for the next stage
    # else, suppressed (set to 0)
# results in a binary image with thin edges

# 4. hysterisis thresholding
# decides which edges are actually edges, and which are not
    # two thresholds minVal and maxVal (select carefully)
    # above maxVal are sure to be edges, below minVal are sure to be non-edges
# in between these minVal and maxVal, classified based on connectivity:
    # if connected to sure-edge pixels, then are also edges
    # else discarded
# also removes small pixel noises by assuming that edges are long lines
# results in strong edges in the image

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("original.png", 0)
edges = cv2.Canny(img, 50, 150)

plt.subplot(121), plt.imshow(img, cmap = 'gray')
plt.title('original'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(edges, cmap = 'gray')
plt.title('edge image'), plt.xticks([]), plt.yticks([])

plt.show()