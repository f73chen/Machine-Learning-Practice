import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt

# read image from OpenCV samples
# optional second argument for image format: 
    # IMREAD_COLOR (-1), IMREAD_GRAYSCALE (0), IMREAD_UNCHANGED (1)
# after reading, stored in a Mat object
img = cv.imread(cv.samples.findFile("starry_night.jpg"), 0)

# doesn't throw error if image path wrong, only returns None
if img is None:
    sys.exit("Could not read the image")

'''
# can create blank window then load an image to it later 
# arg2: if _NORMAL, window can be resized, default _AUTOSIZE
#cv.namedWindow("image", cv.WINDOW_NORMAL)

# arg1: title of the window, arg2: the Mat object
cv.imshow("Display window", img)

# display image forever until user presses key
# returns value of the key pressed
k = cv.waitKey(0) & 0xFF

# save file in working dir if 's' key pressed
if k == ord("s"):
    cv.imwrite("starry_night.png", img)

cv.destroyAllWindows()
# alternatively, use cv.destroyWindow(windowName)
'''

# display image using numpy and matplotlib
# note: image loaded by OpenCV is in BGR mode, but Matplotlib displays in RGB mode
# therefore, colour images displayed incorrectly in matplotlib if read with opencv
# lots of colour map options (recommend 'gray'), default 'rgb'
plt.imshow(img, cmap = 'seismic', interpolation = 'bicubic')

# hide tick values on X and Y axis
plt.xticks([]), plt.yticks([])
plt.show()