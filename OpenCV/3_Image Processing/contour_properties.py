import cv2
import numpy as np

img = cv2.imread("OpenCV.png", 0)
ret, thresh = cv2.threshold(img, 127, 255, 0)

contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

# aspect ratio: ratio of width to height of the bounding rect
x, y, w, h = cv2.boundingRect(cnt)
aspect_ratio = float(w)/h
print("aspect ratio: " + str(aspect_ratio))

# extent: ratio of contour area to bounding rectangle area
area = cv2.contourArea(cnt)
x, y, w, h = cv2.boundingRect(cnt)
rect_area = w * h
extent = float(area)/rect_area
print("extent: " + str(extent))

# solidity: ratio of contour area to convex hull area
area = cv2.contourArea(cnt)
hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)
solidity = float(area)/hull_area
print("solidity: " + str(solidity))

# equivalent diameter: diameter of circle whose area is the same as the contour area
area = cv2.contourArea(cnt)
equi_diameter = np.sqrt(4 * area / np.pi)
print("equivalent diameter: " + str(equi_diameter))

# orientation: angle at which the object is directed
# also gives major axis and minor axis lengths
(x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
print("angle: " + str(angle))

# get all points which comprise the object
mask = np.zeros(img.shape, np.uint8)
img = cv2.drawContours(mask, [cnt], 0, 255, -1)
pixelpoints = np.transpose(np.nonzero(mask))    # give coords in (row, column)
#pixelpoints = np.findNonZero(mask)             # give coords in (x, y)
    # using np method means axes interchanged (col = x, row = y)
    # note: can't print pixel points because they're not image pixels

# max value, min value, and their locations
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img, mask = mask)
print("min val: " + str(min_val) + ", max val: " + str(max_val) + ", min loc: " + str(min_loc) + ", max loc: " + str(max_loc))
    # in this case min and max are the same because the entire mask area is white

# mean colour or mean intensity
mean_val = cv2.mean(img, mask = mask)
print("mean intensity: " + str(mean_val))

# extreme points mean topmost, bottommost, leftmost, and rightmost
leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
print("leftmost: " + str(leftmost) + ", rightmost: " + str(rightmost) + ", topmost: " + str(topmost) + ", bottommost: " + str(bottommost))

# only plots 3 points because leftmost and topmost are redundant
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = cv2.circle(img, leftmost, radius = 2, color = (0, 255, 0), thickness = -1)
img = cv2.circle(img, rightmost, radius = 2, color = (0, 255, 0), thickness = -1)
img = cv2.circle(img, topmost, radius = 2, color = (0, 255, 0), thickness = -1)
img = cv2.circle(img, bottommost, radius = 2, color = (0, 255, 0), thickness = -1)

cv2.imshow("with contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()