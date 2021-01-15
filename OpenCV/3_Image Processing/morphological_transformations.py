# morphological transformations are simple operations based on image shape
    # usually done on binary images
# arg2: structuring element or kernel, which decides the nature of operation

import cv2
import numpy as np

img = cv2.imread('j.png', 0)
kernel = np.ones((9, 9), np.uint8)

# manually created a structuring element, but can do that automatically:
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# [[1, 1, 1, 1, 1],
#  [1, 1, 1, 1, 1],
#  [1, 1, 1, 1, 1],
#  [1, 1, 1, 1, 1],
#  [1, 1, 1, 1, 1]]

kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# [[0, 0, 1, 0, 0],
#  [1, 1, 1, 1, 1],
#  [1, 1, 1, 1, 1],
#  [1, 1, 1, 1, 1],
#  [0, 0, 1, 0, 0]]

kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
# [[0, 0, 1, 0, 0],
#  [0, 0, 1, 0, 0],
#  [1, 1, 1, 1, 1],
#  [0, 0, 1, 0, 0],
#  [0, 0, 1, 0, 0]]

# 1. Erosion
# eordes away boundaries of foreground object (foreground in white)
# kernel slides through the image
    # pixel set to 1 only if all pixels under the kernel is 1, else made 0
# pixels near boundary discarded depending on side of the kernel
# useful for removing small white noise, detach 2 connected objects etc.
#erosion = cv2.erode(img, kernel, iterations = 1)


# 2. Dilation
# pixel element is 1 if at least 1 pixel under the kernel is 1
    # increases white region (foreground object)
# in noise removal, erosion is followed by dilation
    # since noise is gone, it won't come back
# also useful in joining broken parts of an object
#dilation = cv2.dilate(img, kernel, iterations = 1)

# 3. Opening
# another name for erosion followed by dilation
# results in removal of image region boundary pixels
#opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# 4. Closing
# reverse of opening; dilation followed by erosion
# good for closing small holes inside of foreground objects, or small black points on object
# results in filling in of image region boundary pixels
#closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# 5. Morphological Gradient
# difference between dilation and erosion of an image
# looks like the outline of the object
#gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# 6. Top Hat
# difference between input image and opening of the image
#tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# 7. Black Hat
# difference between input image and closing of the image
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

cv2.imshow('new', blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()