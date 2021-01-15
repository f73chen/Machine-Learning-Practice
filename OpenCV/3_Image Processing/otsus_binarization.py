# consider bimodal image (histogram has two peaks)
    # can approx. take value in the middle of the peaks as threshold
# Otsu binarization automatically calculates a thresold from histogram
    # not accurate for images that are not bimodal
# pass threshold value as 0
# if finds optimal threshold value, returns as retVal
    # else, retVal = passed threshold value

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('chess.jpg', 0)

# global thresholding
ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Otsu's thresholding
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# plot images and histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original noisy image', 'histogram', 'global thresholding (v = 127)',
          'Original noisy image', 'historgram', "Otsu's thresholding",
          'Gaussian filtered image', 'histogram', "Otsu's thresholding"]

for i in range(3):
    plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
plt.show()