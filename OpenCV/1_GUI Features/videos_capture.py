import numpy as np
import cv2 as cv

# capture video from the 0th connected camera
cap = cv.VideoCapture(0)

# alternatively, pass the name of a video file
#cap = cv.VideoCapture('qlearn.avi')

while True:
    # capture frame-by-frame
    # also returns True/False if frame is read correctly
        # can check for end of the video by checking return value
    # if cap hasn't initialized the capture, code shows error
        # check with cap.isOpened(), if false open with cap.open()
    ret, frame = cap.read()

    # can access features of the videow ith cap.get(propId) where property ID from 1 - 18
        # each denotes a property of the video
    # ex. check frame width and height with cap.get(3) and cap.get(4)
    # ex. modify width and height: ret = cap.set(3, 50), ret = cap.set(4, 100)
    print(str(cap.get(3)) + " x " + str(cap.get(4)))

    # perform operations on the frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # display the resulting frame
    cv.imshow('frame', gray)

    # change waitKey to change video speed (capture from video file)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

# when finished, release the capture
cap.release()
cv.destroyAllWindows()