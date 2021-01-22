# corners are regions with large intensity variations in all directions
# harris corner detection finds difference in intensity for a displacement of (u, v) in all directions
# E(u, v) = SUM(x, y) w(x, y) * (I(x+u, y+v) - I(x, y))^2
    #     = SUM(x, y) window function * (shifted intensity - intensity)^2
    # window function is either a rectangular window or gaussian window which give weights to pixels underneath
    # goal to maximize function E(u, v), by maximizing second term
# alt. equation:
    # E(u, v) ~~ [u, v] M [u, v]^T, where
    # M = SUM(x, y) w(x, y) [[Ix*Ix, Ix*Iy], [Ix*Iy, Iy*Iy]]
    # where Ix and Iy are image derivatives in x and y directions, found with Sobel
# then use a score based on eigenvalues to decide whether region is corner, edge, or flat
    # L1 and L2 are the eigenvalues of M
    # when |R| small, L1 & L2 small, so region is flat
    # when R < 0, L1 >> L2 or L1 << L2, so region is edge
    # when R large, L1 ~~ L2 and large, so region is a corner
# results in grayscale image with these scores
    # thresholding scores gives corners

import cv2
import numpy as np

img = cv2.imread('box.jpg', -1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

# arg1: input image in grayscale and float32
# arg2: blockSize --> size of neighbourhood considered for corner detection
# arg3: ksize --> aperture parameter for Sobel
# arg4: k --> Harris detector free parameter in equation
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# dilate result to mark corners
dst = cv2.dilate(dst, None)

# threshold for optimal value
#img[dst > 0.01 * dst.max()] = [0, 0, 255]
ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
dst = np.int8(dst)

# find centroids for sub-pixel accuracy
# input: image --> the 8-bit single-channel image to be labeled
# opt input1: connectivity --> 8 for 8-way, 4 for 4-way
# opt input2: ltype --> output image label type (CV_32S or CV_16U)
# opt input3: ccltype --> connected components algorithm type
    # cv.CCL_WU --> SAUF for 8-way, SAUF for 4-way
    # cv.CCL_DEFAULT --> BBDT for 8-way, SAUF for 4-way
    # cv.CCL_GRANA --> BBDT for 8-way, SAUF for 4-way
# output1: labels --> destination labeled image
# output2: stats --> select with (label, COLUMN), where COLUMN is:
    # CC_STAT_LEFT --> the leftmost x coordinate which is the inclusive start of the bounding box in the horizontal direction
    # CC_STAT_TOP --> the topmost y coordinate which is the inclusive start of the bounding box in the vertical direction
    # CC_STAT_WIDTH --> the horizontal size of the bounding box
    # CC_STAT_HEIGHT --> the vertical size of the bounding box
    # CC_STAT_AREA --> total area (in pixels) of the connected component
# output3: centroids --> centroid for each label, including the background label
    # accessed via centroids(label, 0) for x and centroids(label, 1) for y
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# define criteria to stop and refine corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

# draw the corners
res = np.hstack((centroids, corners))
res = np.int0(res)
img[res[:, 1], res[:, 0]] = [0, 0, 225]
img[res[:, 3], res[:, 2]] = [0, 255, 0]

cv2.imshow('dst', img)
cv2.waitKey(0)
cv2.destroyAllWindows()