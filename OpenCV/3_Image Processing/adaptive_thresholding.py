# adaptive thresholding calculates the threshold for small regions of the image
    # different thresholds for different regions
    # good for images with varying illumination
# adaptive methods (how the threshold is calculated)
    # cv2.ADAPTIVE_THRESH_MEAN_C --> threshold value is mean of neighbourhood area
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C --> threshold value is weighted sum of neighbourhood values, where weights are a gaussian window
# block size --> size of neighbourhood area
# C --> constant subtracted from calculated mean or weighted sum

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sudoku.jpg", 0)
img = cv2.medianBlur(img, 5)

ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

titles = ['Original image', 'Global threshholding (v = 127)', 'Adaptive mean thresholding', 'Adaptive gaussian thresholding']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()