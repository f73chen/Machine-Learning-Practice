# use cv2.cvtColor(input_image, flag)
    # where flag determines the type of conversion
# ex. BGR to Gray use cv2.COLOR_BGR2GRAY
# ex. BGR to HSV use cv2.COLOR_BGR2HSV
    # note: different softwares use different scales for HSV, so if comparint OpenCV values, need to normalize ranges

# take each frame of the video
# convert from BGR to HSV colour-space
# threshold the HSV image for a range of blue colour
# then extract the blue object alone, then do whatever to the image

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):
    # take each frame
    _, frame = cap.read()

    # convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue in HSV
    lower_blue = np.array([105, 30, 30])
    upper_blue = np.array([135, 255, 255])
    
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([15, 255, 255])
    
    lower_green = np.array([45, 30, 30])
    upper_green = np.array([75, 255, 255])

    # threshold HSV to only get region with blue colours
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # combine all three masks into one
    mask = cv2.add(mask_blue, mask_red)
    mask = cv2.add(mask, mask_green)

    # bitwise AND the mask and the original image
        # regions without blue becomes black (0)
        # regions with blue retains original colour
    res = cv2.bitwise_and(frame, frame, mask = mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask_blue', mask_blue)
    cv2.imshow('mask_red', mask_red)
    cv2.imshow('mask_green', mask_green)
    cv2.imshow('mask_green', mask)
    cv2.imshow('res', res)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()