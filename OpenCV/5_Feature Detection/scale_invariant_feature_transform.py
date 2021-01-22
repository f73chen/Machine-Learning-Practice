# NOTE: THIS PACKAGE IS NOT INCLUDED IN FREE OPENCV

# harris and shi-tomasi corner detectors are rotation-invariant
# however, corner may look more like a curve/edge when scaled

# 1. scale-space extrema detection
# Laplacian of [or?] Gaussian is found for image w/ various sigma values
# LoG: detects blobs in various sizes due to change in sigma
    # sigma acts as a scaling parameter
    # low sigma = smaller corner; high sigma = larger corner
# find local maxima across the scale and space, which gives list of (x, y, sigma)
    # there is a potential keypoint at (x, y) at sigma scale
# LoG costly, so use Difference of Gaussians (DoG) for approx.
    # difference of Gaussian blurring of an image with two different sigma
    # done for different octaves of the image in Gaussian pyramid
# once DoG found, images searched for local extrema over scale and space
    # one pixel in image is compared with its 8 neighbours and 9 pixels in next & previous scale (3d box)
    # if local extrema, then is potential keypoint
    # aka the keypoint is best represented in that scale

# 2. keypoint localization
# refine found keypoints to get more accurate results
# use Taylow series expansion of scale space for more accurate location of extrema
    # rejected if intensity at this extrema is less than threshold
# DoG has higher response for edges, so also remove edges
    # use 2x2 Hessian matrix to compute principal curvature
    # from harris corner detection know that for edges, one eigenvalue is larger than the other
    # if ratio > edgeThreshold, discard that keypoint
# therefore, eliminates low-contrast keypoints and edge keypoints
    # remaining are strong interest points

# 3. orientation assignment
# assign orientation to achieve rotation invariance
# neighbourhood taken around keypoint depending on scale
    # find gradient mag and dir
# create orientation histogram w/ 36 bins for 360 degrees
    # highest peak in histogram and any peak above 80% considered to calculate orientation
# creates keypoints w/ same location and scale, but different directions
    # contributes to stability of matching

# 4. keypoint descriptor
# 16 * 16 neighbourhood around keypoint taken
    # divided into 16 sub-blocks of 4x4 size
    # for each sub-block, create 8 bin orientation histogram
    # 128 available bin values in total
# add other measures for robustness against ilumination, rotation changes etc.

# 5. keypoint matching
# match keypoints between 2 images by identifying their nearest neighbours
# however, second-closest match may be near to first due to noise etc.
    # then use ratio of closest-distance to second-closest distance
    # rejected if > 0.8
    # good at eliminating false matches and retaining good ones

import cv2
import numpy as np

img = cv2.imread('chess.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp = sift.detect(gray, None)

img = cv2.drawKeypoints(gray, kp)

cv2.imshow('keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()