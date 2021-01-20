# in grayscale image, high intensity denotes peaks and hills;
    # low intensity denotes valleys
# start filling isolated valleys (local minima) with different coloured water (labels)
# as water rises, different valleys begin to merge
    # put barriers where water merges
# when peaks are covered, barriers give segmentation result

# to avoid oversegmentation, OpenCV has marker-based algo
    # have to specify which are valley points to be merged and which are not
# 1. label region sure of being the foreground or object with one colour/intensity
# 2. label region sure of being background or non-object with another
# 3. label 0 to region not sure of anything (marker)
# marker will be updated with given labels
# boundaries get a value of -1

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('coins.jpg')
#img = cv2.imread('original.png')
#b, g, r = cv2.split(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#thresh = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# know that region near center are foreground and region away from objects are background
# not sure about boundary regions
# remove white noise and boundary pixels with erosion
    # sure that whatever remaining is a coin
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

# dilate to increase boundary, so sure that background really is background
sure_bg = cv2.dilate(opening, kernel, iterations = 3)

# threshhold distance transform to find sure foreground
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# find unknown region
# subtract foreground object centers from background dilated blobs
    # get non-overlapped border region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# create marker as array of same size as original image, but int32
    # label known regions with positive but different integers
    # label unknown regions with 0
# cv2.connectedComponents() labels background with 0,
    # and other objects with integers starting from 1
    # arrange labels to be what watershed expects
ret, markers = cv2.connectedComponents(sure_fg)

# change background from 0 to 1
markers = markers + 1

# mark unknown regions with 0
markers[unknown == 255] = 0

# now with seeded objects and background, apply watershed to find borders
# note: copied markers, else markers itself will be modified
water = cv2.watershed(img, markers.copy())
img[water == -1] = [255, 0, 0]


plt.subplot(241), plt.imshow(thresh)
plt.title('threshold'), plt.xticks([]), plt.yticks([])

plt.subplot(242), plt.imshow(opening)
plt.title('opening'), plt.xticks([]), plt.yticks([])

plt.subplot(243), plt.imshow(sure_bg)
plt.title('background'), plt.xticks([]), plt.yticks([])

plt.subplot(244), plt.imshow(dist_transform)
plt.title('distance transform'), plt.xticks([]), plt.yticks([])

plt.subplot(245), plt.imshow(sure_fg)
plt.title('foreground'), plt.xticks([]), plt.yticks([])

plt.subplot(246), plt.imshow(markers)
plt.title('markers'), plt.xticks([]), plt.yticks([])

plt.subplot(247), plt.imshow(water)
plt.title('watershed'), plt.xticks([]), plt.yticks([])

plt.show()