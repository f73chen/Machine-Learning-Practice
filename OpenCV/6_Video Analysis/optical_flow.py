# optical flow is the pattern of apparent motion of object between 2 frames
    # caused by the movement of object or camera
# 2D vector field where each vector is a displacement vector of points
# assumptions:
    # the pixel intensities of an object do not change between consecutive frames
    # neighbouring pixels have similar motion
# consider pixel I(x, y, t) which moves by (dx, dy) after dt time
    # since pixels and intensities are the same,
    # I(x, y, t) = I(x + dx, y + dy, t + dt)
# take taylor seires approx. of RHS, remove common terms, and divide by dt
    # fx*u + fy*v + ft = 0
    # fx = df/dx, fy = df/dy, u = dx/dt, v = dy/dt
# can find fx and fy since they're image gradients
    # ft is the gradient along time
    # but u, v are unknown

# Lucas-Kanade method
# assumes 3x3 patch around the point all have the same motion
    # can find fx, fyt, ft
    # use least square fit to find optimal solution
# harder to track if there are large motions, so use pyramids to reduce motion size

# decide good points to track with cv2.goodFeaturesToTrack()
# take first frame, decide Shi-Tomasi corner points, then iteratively track with cv2.calcOpticalFlowPyrLK
    # pass previous frame, previous points, and next frame
    # returns next points and status numbers (1 if next point found, else 0)
    # iteratively pass points as previous points in next step

import cv2
import numpy as np

cap = cv2.VideoCapture('slow.flv')
#cap = cv2.VideoCapture(0)

# params for ShiTomasi corner detection
feature_params = dict(maxCorners = 100,
                      qualityLevel = 0.3,
                      minDistance = 7,
                      blockSize = 7)

# params for lucas kanade optical flow
lk_params = dict(winSize = (15, 15),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# create random colours for the markers
colour = np.random.randint(0, 255, (100, 3))

# take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# create mask image for drawing (same shape and type as original)
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
        # next points --> 2D points w/ new positions of input features in the second image
        # status --> each element of vector set to 1 if flow for features has been found, else 0
        # err --> each element of vector set to error for that point, undef if flow not found
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # select good points (st == 1 means next flow has been found)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), colour[i].tolist(), 2)    # the trailing line
        frame = cv2.circle(frame, (a, b), 5, colour[i].tolist(), -1)    # the current point
    img = cv2.add(frame, mask)

    cv2.imshow('frame', img)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # update previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2) # [[[x1, y1]], [[x2, y2]], ...]

cv2.destroyAllWindows()
cap.release()