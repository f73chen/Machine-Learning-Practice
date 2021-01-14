# img --> the image to draw the shapes
# color --> colour of the shape
    # For BGR, pass as tuple (255, 0, 0). For grayscale, pass scalar value
# thickness --> thickness of the line. Default = 1
    # if -1 for enclosed shapes, fills the whole shape
# lineType --> whether 8-connected (default), anti-aliased etc.
    # cv2.LINE_AA looks great for curves

import numpy as np
import cv2 as cv

# create a black image
img = np.zeros((512, 512, 3), np.uint8)

# draw diagonal blue line with thickness 5px from bottom left to top right
img = cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)

# draw rectangle from top left to bottom right
img = cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)

# draw circle with center coordinates and radius
img = cv.circle(img, (447, 63), 63, (0, 0, 255), -1)

# draw ellipse with center location, axes lengths, angle ccw, startAngle and endAngle of arc cw from major axis
img = cv.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)

# draw polygon with arrays of coordinates of vertices
# if arg3 = False, get polylines joining all points, not a closed shape
# for drawing multiple lines, better to call polylines() than line() multiple times
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
pts = pts.reshape((-1, 1, 2))
img = cv.polylines(img, [pts], True, (0, 255, 255))

# add text with:
    # text data to write
    # position coordinates of where to put it
    # font type
    # font scale/size
    # color, thickness, lineType
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv.LINE_AA)

cv.imshow("Display window", img)
k = cv.waitKey(0) & 0xFF
cv.destroyAllWindows()