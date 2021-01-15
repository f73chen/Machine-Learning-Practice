# if the pixel value is greater than a threshold value, assigned some value
    # else assigned some other value
# arg1: source image (in grayscale)
# arg2: threshold valued used to classify pixels
# arg3: value to be given if pixel value passes threshold
# arg4: style of thresholding
    # cv2.THRESH_BINARY --> maxVal if src(x, y) > thresh, else 0
    # cv2.THRESH_BINARY_INV --> 0 if src(x, y) > thresh, else maxVal
    # cv2.THRESH_TRUNC --> threshold if src(x, y) > thresh, else src(x, y)
    # cv2.THRESH_TOZERO --> src(x, y) if src(x, y) > thresh, else 0
    # cv2.THRESH_TOZERO_INV --> 0 if src(x, y) > thresh, else src(x, y)

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("OpenCV.png", 0)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()