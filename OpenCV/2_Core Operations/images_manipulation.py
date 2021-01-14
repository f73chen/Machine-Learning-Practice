import cv2
import numpy as np

img = cv2.imread(cv2.samples.findFile("starry_night.jpg"), -1)
px = img[100, 100]
print(px)   # [131  54  22]

# only print blue pixel
blue = img[100, 100, 0]
print(blue) # 131

# can modify pixel values
img[100, 100] = [255, 255, 255]
print(img[100, 100])    # [255, 255, 255]

# in actual practice, numpy array.item() and array.itemset() is faster
    # however, only returns a scalar
    # so call array.item() separately for each B, G, R
img.itemset((10, 10, 2), 100)
img.item(10, 10, 2)     # 100

# image properties include number of rows, columns, and channels, type of image data, number of pixels etc.
print(img.shape)        # tuple of rows, columns, and channels (if grayscale only returns rows and columns)
print(img.size)         # total number of pixels
print(img.dtype)        # datatype (very important for debugging)

# obtain subset region of image (ROI) using numpy
tree = img[100:300, 200:400]
img[200:400, 0:200] = tree

# split image into the three channels then merge back together
b, g, r = cv2.split(img)
# or b = img[:, :, 0]
img = cv2.merge((b, g, r))

# make all red pixels black
# numpy indexing is much more efficient than cv2.split
img[:, :, 2] = 0

cv2.imwrite("starry_night.png", img)