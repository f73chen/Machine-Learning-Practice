# used for image segmentation or finding objects
# creates an image of the same size (but single channel) as the input image
# each pixel corresponds to the probability of that pixel belonging to the object
    # object of interest presented in more white than the rest
# used with camshift algorithm etc.

# create histogram of image containing the object
# object should fill the image as far as possible for better results
# colour image defines better than grayscale
# resulting probabilities on thresholding should give the object alone

# first calculate colour histogram of both the object (M) and image (I)
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ROI: object or region of interest
roi=  cv2.imread("rose_red.jpg")
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# target is the image to search in
target = cv2.imread("rose.jpg")
hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

# find histograms using calcHist
M = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
I = cv2.calcHist([hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])
I[I == 0] = 0.0001
R = M/I

# find ratio R = M/I then backproject R
# create new image with every pixel as probability of being target
# B(x, y) = R[h(x, y), s(x, y)] 
    # h = hue, s = saturation at (x, y)
    # B(x, y) = min[B(x, y), 1]
h, s, v = cv2.split(hsvt)
B = R[h.ravel(), s.ravel()]
B = np.minimum(B, 1)
B = B.reshape(hsvt.shape[:2])

# apply convolution with circular disk B = D * B, where D is the disk kernel
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cv2.filter2D(B, -1, disc, B)
B = np.uint8(B)
cv2.normalize(B, B, 0, 255, cv2.NORM_MINMAX)

# location of max intensity gives location of the object
ret, thresh = cv2.threshold(B, 50, 255, 0)
thresh = cv2.merge((thresh, thresh, thresh))
res = cv2.bitwise_and(target, thresh)

cv2.imshow("result", res)
cv2.waitKey(0)
cv2.destroyAllWindows()