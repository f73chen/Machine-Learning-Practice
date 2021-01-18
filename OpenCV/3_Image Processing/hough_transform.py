import cv2
import numpy as np

img = cv2.imread('sudoku.jpg')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(imgray, 50, 150, apertureSize = 3)

'''
# hough lines finds lines by voting on rho and theta combinations
# arg1: input (binary) image from threshold or canny edge detection
# arg2: rho --> perpendicular distance from origin to the line
# arg3: theta --> angle of perpendicular line and horizontal axis, CCW
# arg4: threshold --> minimum vote to be considered a line
    # number of votes depens on number of points on the line
    # represents the minimum length of line that should be detected
lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
for line in lines:
    rho, theta = line.flatten()
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
'''

# probabilistic model directly returns x, y start/end coordinates
# lower computation, but requires lower threshold
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
for line in lines:
    x1, y1, x2, y2 = line.flatten()
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Hough lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()