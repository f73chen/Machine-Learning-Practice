# in SIFT, approximated Laplacian of Gaussian with difference of gaussian for scale-space
# SURF approximates LoG with box filter
# better b/c convolution w/ box filter can be easily calculated with integral images
    # can be done in parallel for different scales
# relies on determinant of Hessian matrix for both scale and location

# for orientation assignment, SURF uses wavelet responses in hor. & vert. direction for neighbourhood of size 6
# dominant orientation estimated by calculating sum of all responses within sliding orientation window of 60 degrees
    # if {upright} = 0, calculate orientation. If 1, don't calculate (faster)

# take into account the sign of the Laplacian
# adds no computation cost (already computed during detection)
# sign distinguishes bright on dark from dark on bright
# only compare features if they have the same type of contrast
# faster matching without reducing performance
# summary: adds lots of features to improve speed (3x faster than SIFT)
# good at blurring and rotation, but not viewpoint and illumination change
# basically a blob detector

import cv2
import numpy as np

img = cv2.imread("box.jpg", 0)
surf = cv2.SURF(400)    # Hessian threshold = 400
keypoints, descriptors = surf.detectAndCompute(img, None)
print(len(keypoints))

# Note: can't use it b/c don't have SURF installed???