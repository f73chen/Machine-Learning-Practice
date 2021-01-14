import cv2
import numpy as np

def nothing(x):
    pass

# create a black image in a window
img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('image')

# create trackbars for colour change
    # arg1 --> trackbar name
    # arg2 --> the window to which it is attached
    # arg3 --> default value
    # arg4 --> max value
    # arg5 --> callback function when trackbar value changes
cv2.createTrackbar('R', 'image', 0, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('B', 'image', 0, 255, nothing)

# create switch for ON/OFF functionality
# note: OpenCV doesn't have a button functionality, so use trackbar instead
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image', 0, 1, nothing)

while(1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        break

    # get current positions of the 4 trackbars
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    s = cv2.getTrackbarPos(switch, 'image')

    # if switch off, only show black screen
    if s == 0:
        img[:] = 0

    # else display the selected colours
    else:
        img[:] = [b, g, r]

cv2.destroyAllWindows()