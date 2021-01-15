# normally, work with image of constant size
# sometimes, need different resolutions of the same image
# need to create a set of images w/ different resolution and search for object in all images
    # called image pyramids, because kept in ordered stack

# 1. Gaussian pyramid
# higher level (low resolution) images in Gaussian pyramid formed by 
    # removing consecutive rows and columns in lower level (higher res image)
# each pixel in higher level formed by contribution from 5 pixels in underlying level with gaussian weights
    # kernel has gaussian/normal distribution
# so, a M*N image becomes M/2 * N/2
    # area reduced to 1/4
# similarily, when expanding, area becomes 4x in each level

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('OpenCV.png', 0)
lowres = cv2.pyrDown(img)
highres = cv2.pyrUp(img)
restore = cv2.pyrUp(lowres)

plt.subplot(221), plt.imshow(img)
plt.title('original'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(lowres)
plt.title('low res'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(highres)
plt.title('high res'), plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.imshow(restore)
plt.title('restored'), plt.xticks([]), plt.yticks([])

plt.show()

# Laplacian pyramids are formed form Gaussian pyramids
    # like edge images only w/ mostly 0's
    # used in image compression
# a level is formed by the difference between that level in Gaussian pyramid
    # and the expanded version of its upper level in gaussian pyramid