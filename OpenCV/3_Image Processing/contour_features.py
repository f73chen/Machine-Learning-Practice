# image moments used to calculate center of mass, area etc.

import cv2
import numpy as np

img = cv2.imread("capture.png", 0)
ret, thresh = cv2.threshold(img, 127, 255, 0)

# recall: contours is a list, or tree of lists of points
# hierarchy is how the shapes relate to each other
    # ex. if they're on top of each other
# note: cv2.findContours eats the image while finding contours,
    # so need to copy it to print the contours in colour
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#print(contours[0])
#print(hierarchy[:3])

cnt = contours[0]
M = cv2.moments(cnt)
#print(M)

# from moments, extract data like area, centroid etc.
    # Cx = M10/M00, Cy = M01/M00
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

# contour area from cv2.contourArea() or from moment M['m00']
area = cv2.contourArea(cnt)
#print(int(area) == int(M['m00']))  # True

# contour perimeter, aka arc length
    # arg2 specifies whether shape is a closed contour or just a curve
perimeter = cv2.arcLength(cnt, True)

# contour approx. approximates a contour to another shape with fewer vertices
    # epsilon --> max distance from contour to approx. contour (accuracy)
    # need to choose epsilon carefully
# returns another set of (x, y) points
epsilon = 0.1 * perimeter
approx = cv2.approxPolyDP(cnt, epsilon, True)

# convex hull checks curve for convexity defects and corrects it
# convex curves are always bulged out or at least flat
    # bulge inside = convexity defect
# points --> contours to pass into
# hull --> output, usually avoid it
# clockwise --> if true, then oriented clockwise
# returnPoints --> if true, returns coordinates of the hull points
    # else returns indices of contour points corresponding to hull points
#hull = cv2.convexHull(points[, hull[, clockwise[, returnPoints]]
#hull = cv2.convexHull(cnt, returnPoints = False)   
    # note: can't drawContour if returnPoints = False
hull = cv2.convexHull(cnt)
#print(hull)

# check if a curve is convex or not
    # returns true or alse
k = cv2.isContourConvex(cnt)
#print(k)

# straight bounding rectangle: doesn't consider rotation of object
# let (x, y) be top-left coordinates and (w, h) be width and height
x, y, w, h = cv2.boundingRect(cnt)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# rotated rectangle draws bounding box with minimum area
# returns top-left coner (x, y), (width, height), angle of rotation
# but to draw need to obtain 4 corners from cv2.boxPoints()
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
#img = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

# the minimum enclosing circle completely covers object with minimum area
(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
#img = cv2.circle(img, center, radius, (0, 255, 0), 2)

# fit ellipse: returns rotated rectangle in which the ellipse is inscribed
# note: doesn't necessarily completely cover object
ellipse = cv2.fitEllipse(cnt)
#img = cv2.ellipse(img, ellipse, (0, 255, 0), 2)

# approximate fitting a straight line
# returns a parallel normalized vector (vx, vy) and a point on the line (x, y) 
rows, cols = img.shape[:2]
[vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0, 0.01, 0.01)
# calculate y-intercepts with the image boundaries
lefty = int((-x * vy / vx) + y)
righty = int(((cols - x) * vy / vx) + y)
img = cv2.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

# note: hull must be passed as an array, else each point is treated as a contour
#img = cv2.drawContours(img, [hull], -1, (0, 255, 0), 3)
cv2.imshow("with contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()