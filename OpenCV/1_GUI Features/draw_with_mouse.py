import cv2 as cv
import numpy as np

# True if mouse is pressed
drawing = False

# if True, draw rectangles, else press 'm' to toggle to curve
mode = True

ix, iy = -1, -1

# mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    # set first coordinate parameter
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv.circle(img, (x, y), 5, (0, 0, 255), -1)

    # when mouse released, set drawing = false
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        else:
            cv.circle(img, (x, y), 5, (0, 0, 255), -1)


# create a black image, a window, and bind the function to the window
img = np.zeros((512, 512, 3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image', draw_circle)

while(1):
    cv.imshow('image', img)
    k = cv.waitKey(20) & 0xFF

    # toggle between rectangle and circle mode
    if k == ord('m'):
        mode = not mode
    elif k == ord('q'):
        break

cv.destroyAllWindows()