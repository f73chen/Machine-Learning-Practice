import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# define the codec and create the VideoWriter object
#fourcc = cv.VideoWriter_fourcc('D', 'I', 'V', 'X')
fourcc = cv.VideoWriter_fourcc(*'DIVX')

# arg1: output file name
# arg2: fourcc code --> 4-byte code used to specify the video codec
# arg3: number of frames per second
# arg4: frame size
# arg5: if true, expects colour frame, else works with grayscale
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480), isColor = True)

while(cap.isOpened()):
    ret, frame = cap.read()

    # if camera is capturing images
    if ret == True:
        # flip the frame in the vertical direction
        frame = cv.flip(frame, 0)

        # VideoWriter object write the flipped frame
        out.write(frame)

        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# release everything when job is finished
cap.release()
out.release()
cv.destroyAllWindows()