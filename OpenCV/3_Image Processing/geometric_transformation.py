# cv2.warpAffine takes 2x3 transformation matrix
# cv2.warpPerspective takes 3x3 transformation matrix
# cv2.resize() --> resize image
    # specify size of resultant image or scaling factor
# resize interpolation methods:
    # cv2.INTER_AREA  --> shrinking
    # cv2.INTER_CUBIC --> zooming (slow)
    # CV2.INTER_LINEAR --> zooming (used by default for all resizing)

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('chess.jpg', 0)
rows, cols = img.shape

# in perspective transformation, straight lines remain straight (but not necessarily parallel)
# need 3x3 transformation matrix
# 4 points on input and corresponding points on output
    # 3/4 points should not be collinear
# generate transformation matrix with cv2.getPerspectiveTransform
pts1 = np.float32([[56, 65], [500, 50], [30, 400], [250, 430]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

M = cv2.getPerspectiveTransform(pts1, pts2)
res = cv2.warpPerspective(img, M, (300, 300))

plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(res), plt.title('output')
plt.show()

'''
# rotate using transformation matrix M of the form:
    # [[cos(t), -sin(t)], [sin(t), cos(t)]]
# OpenCV provides scaled rotation with adjustable center of rotation
    # [[a, b, (1-a)*center.x - b*center.y], [-b, a, b*center.x + (1-a)*center.y]]
    # a = scale * cos(t)
    # b = scale * sin(t)
M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)    # rotates by 90* about center without scaling
res = cv2.warpAffine(img, M, (cols, rows))
'''

'''
# in affine transformation, all parallel lines in original image remain parallel
# to find transformation matrix, need 3 points from input image and corresponding locations in output
    # cv2.getAffineTransform generates the 2x3 matrix
img = cv2.imread('chess.jpg')
rows, cols, ch = img.shape

pts1 = np.float32([[50, 50], [100, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

M = cv2.getAffineTransform(pts1, pts2)
res = cv2.warpAffine(img, M, (cols, rows))

plt.subplot(121), plt.imshow(img), plt.title('input')
plt.subplot(122), plt.imshow(res), plt.title('output')
plt.show()
'''

'''
# M with shape [[1, 0, tx], [0, 1, ty]]
    # tx --> translation in x direction
    # ty --> translation in y direction
M = np.float32([[1, 0, 100], [0, 1, 50]])   # shifts 100 right and 50 down

# arg3: size of output image in form (width, height) = (cols, rows)
res = cv2.warpAffine(img, M, (cols, rows))
'''

'''
res = cv2.resize(img, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)

# or:

height, width = img.shape[:2]
res = cv2.resize(img, (2*width, 2*height), interpolation = cv2.INTER_CUBIC)
'''

'''
cv2.imshow('img', res)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''