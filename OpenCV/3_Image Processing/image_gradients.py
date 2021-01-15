# Sobel is a joint Gaussian smoothing + differentiation operation
    # more resistant to noise
# specify direction of derivatives: vertical or horizontal by yorder, xorder
# specify size of kernel by ksize
    # if ksize = -1, a 3x3 Scharr filter is used which is better than a 3x3 Sobel filter

# Laplacian derivatives calculate the laplacian of the image given by:
    # delta(src) = d(src)^2/dx^2 + d(src)^2/dy^2
    # where each derivative is found using Sobel derivatives
# if ksize = -1, then use the following kernel:
    # 0  1  0
    # 1 -4  1
    # 0  1  0

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('capture.png', 0)

'''
laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)

plt.subplot(2, 2, 1), plt.imshow(img, cmap = 'gray')
plt.title('original'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap = 'gray')
plt.title('sobelx'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4), plt.imshow(sobely, cmap = 'gray')
plt.title('sobely'), plt.xticks([]), plt.yticks([])
'''

# problem is, black to which taken as positive slope, and white to black has negative slope
# when convert data into np.uint8, all negative slopes are made 0
    # therefore miss that edge
# to detect both edges, keep the output datatype to some higher form
    # ex. cv2.CV_16S, cv2.CV_64F
    # take abs value, then convert back to cv2.CV_8U

# output dtype = cv2.CV_8U
sobelx8u = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize = 5)

# output dtype = cv2.CV_64
# then take absolute and convert back to cv2.CV_8U
sobelx64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

plt.subplot(1, 3, 1), plt.imshow(img, cmap = 'gray')
plt.title('original'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2), plt.imshow(sobelx8u, cmap = 'gray')
plt.title('sobel CV_8U'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 3), plt.imshow(sobel_8u, cmap = 'gray')
plt.title('sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])

plt.show()