import cv2
import numpy as np

img = cv2.imread("contour.png")
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)

contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

# deviations from convex hull is a convexity defect
    # note: must pass returnPoints = False
# returns array where each row contains: start point, end point, farthest point, approx. distance to farthest point
# draw a line joining start point and end point, then draw dot at farthest point
hull = cv2.convexHull(cnt, returnPoints = False)
defects = cv2.convexityDefects(cnt, hull)

for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(img, start, end, [0, 255, 0], 2)   # draws closest boundary segment
    cv2.circle(img, far, 5, [0, 0, 255], -1)    # draws the defect point

# point polygon test finds shortest distance between point and a contour
# distance negative when point outside contour, pos inside, 0 on contour
# arg3: measureDist: if true then find signed distance
    # if false find whether point is inside, outside, or on contour (+1, -1, 0)
    # if don't care about distance, using False speeds up by 2-3X
dist = cv2.pointPolygonTest(cnt, (50, 50), True)
print("dist: " + str(dist))

cv2.imshow("with contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()