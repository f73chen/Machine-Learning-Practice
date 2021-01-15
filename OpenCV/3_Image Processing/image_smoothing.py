# low pass filter (LPF) remove noise or blur the image
    # removes high frequency content (noise, edges)
# high pass filter (HPF) find edges in an image
# in a 5x5 averaging filter kernel:
    # for each pixel, a 5x5 window is centered on this pixel
    # all pixels falling within this window are summed up
    # result divided by 25
    # performed for all pixels in the image

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('sudoku.jpg')
row, col, ch = img.shape

# 1. blurring via averaging
# convolve the image with a normalized box filter
# takes average of all pixels under kernel area and replaces central element with the average
    # if not using normalized box filter, pass normalize=False to cv2.boxFilter()
#blur = cv2.blur(img, (5, 5))

# 2. gaussian filtering
# use Gaussian kernel instead of box filter with equal filter coefficients
# specify width and height of the kernel, standard deviation in X and Y directions, and sigmaX, sigmaY
    # if only give sigmaX, assume sigmaY = sigmaX
    # if both are 0, calculated from kernel size
# gaussian noise is where the noise deviations are gaussian-distributed (follows probability density function)
# note: may be replaced by some value not in the original image
#blur = cv2.GaussianBlur(img, (5, 5), 0)

# 3. median filtering
# computes median of all pixels under the kernel window
    # replace central pixel with this value
# effective at removing salt-and-pepper noise
# resultant pixel value must exist in original image
# kernel size must be positive odd integer
#median = cv2.medianBlur(img, 5)

# 4. bilaterial filtering
# highly effective at noise removal while preserving edges
# however, slower compared to others
# caussian filter is a function of space along
    # only considers nearby picels
    # not whether pixels have almost the same intensity
    # doesn't consider whether pixel lies on an edge or not
# bilateral also uses Gaussian filter in space, 
    # but uses one more (multiplicative) Gaussian filter as a function of pixel intensity differences
    # only pixels which are spatial neighbours are considered for filtering
    # only pixels with intensities similar to the entral pixel are included to compute the blurred intensity value
    # for pixels lying near edges, neighbouring pixels on the other side of the edge (large intensity variations) will not be included for blurring
# note: surface texture is gone, but edges are preserved
blur = cv2.bilateralFilter(img, 9, 75, 75)

plt.subplot(121), plt.imshow(img), plt.title('original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('blurred')
plt.xticks([]), plt.yticks([])
plt.show()

'''
# 2D convolution/filtering
kernel = np.ones((5, 5), np.float32)/25
dst = cv2.filter2D(img, -1, kernel)

plt.subplot(121), plt.imshow(img), plt.title('original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('averaging')
plt.xticks([]), plt.yticks([])
plt.show()
'''