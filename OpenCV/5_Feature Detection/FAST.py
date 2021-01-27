# high-speed corner detector
# 1. select pixel p in image as an interest point w/ intensity Ip
# 2. select threshold value t
# 3. consider circle of 16 pixels under test
# 4. pixel p is a corner if there are n contiguous pixels in the circle of 16
    # which are all brighter than Ip + t or darker than Ip - t
# 5. high-speed test to exclude non-corners:
    # examines only pixels at 1, 9, 5, 13
    # if p is a corner, then at least 3 of these must all be brighter or darker
    # else p cannot be a corner
    # if passed, examine the rest of the pixels
# 5. b. weaknesses:
    # doesn't reject as many candidates for n < 12
    # non-optimal choice of pixels because efficiency depends on ordering of the questions and distribution of corner appearances
    # results of high-speed tests are thrown away
    # multiple features are detected adjacent to one another

# machine learning a corner detector
# 1. select set of images (perferably from target application domain)
# 2. run FAST algo in every image to find feature points
# 3. for each feature point, store the 16 pixels as a vector
    # get featrure vector P
# 4. each of the 16 pixels can be d(arker), s(imilar), or b(righter)
# 5. subdivide P into Pd, Ps, Pb
# 6. boolean Kp is true if p is a corner and false otherwise
# 7. use ID3 decision tree classifier to select the x which yields the most info
    # query each subset using var Kp for knowledge about the true class
    # about whether the candidate pixel is a corner, measured by entropy Kp
# 8. recursively applied until entropy = 0
# 9. create decision tree for fast detection in other images

# non-maximal suppression
# prevent detecting multiple interest points in adjacent locations
# 1. compute score function V for all detected feature points
    # V = sum of absolute difference between p and 16 surrounding pixel values
# 2. consider 2 adjavent keypoints and compute their V
# 3. discard the one with lower V

# summary:
# faster than other existing corner detectors
# however, not robust to high levels of noise, and dependent on a threshold

# can specify threshold, whether non-max suppression, neighbourhood etc.
# neighbourhood flags:
    # cv2.FAST_FEATURE_DETECTOR_TYPE_5_8
    # cv2.FAST_FEATURE_DETECTOR_TYPE_7_12
    # cv2.FAST_FEATURE_DETECTOR_TYPE_9_16

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("box.jpg", 0)
# initiate FAST object with default values

fast = cv2.FastFeatureDetector_create()

# find and draw keypoints
kp = fast.detect(img, None)
img2 = cv2.drawKeypoints(img, kp, outImage = None, color = (255, 0, 0))

# print all defauly params
print(f"Threshold: {fast.getInt('threshold')}")
print(f"nonmaxSuppression: {fast.getBool('nonmaxSuppression')}")
print(f"neighbourhood: {fast.getInt('type')}")
print(f"total keypoints with nonmaxSuppression: {len(kp)}")

cv2.imshow('fast_true', img2)

# disable nonmaxSuppression
fast.setBool('nonmaxSuppression', 0)
kp = fast.detect(img, None)
print(f"total keypoints without nonmaxSuppression: {len(kp)}")

img3 = cv2.drawKeypoints(img, kp, color = (255, 0, 0))
cv2.imshow('fast_false', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Note: don't think this code is compatible with my version of OpenCV