# consider set of points with a small window (ex. window)
# have to move that window to area of max pixel density (max number of points)
# initialize first window, then find centroid of all points inside
    # move new window to 1st window's centroid
    # continue until center and centroid in same place or close enough
# so obtain a window with max pixel distribution

# recall histogram backprojection, where a sample/slice of the object is used to get object historgram distribution
    # object histogram then matches image histogram
    # therefore separates object from background

import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('slow.flv')

# record first frame of the video
ret, frame = cap.read()

# set up initial location of window
y, h, x, w = 200, 40, 320, 40  # hardcoded for slow.flv
track_window = (x, y, w, h)

# set of Region of Interest for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# remove areas of low light (note HSV form)
mask = cv2.inRange(hsv_roi, np.array((0., 0., 127.)), np.array((180., 255., 255.)))

# inputs: images, channels, mask image, histSize, ranges
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

# inputs: source array, destination array, min, max, format
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# termination criteria: either 10 iterations or move less than 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# plt.subplot(231), plt.imshow(frame)
# plt.title('frame'), plt.xticks([]), plt.yticks([])
# plt.subplot(232), plt.imshow(roi)
# plt.title('roi'), plt.xticks([]), plt.yticks([])
# plt.subplot(233), plt.imshow(hsv_roi)
# plt.title('hsv_roi'), plt.xticks([]), plt.yticks([])
# plt.subplot(234), plt.imshow(mask)
# plt.title('mask'), plt.xticks([]), plt.yticks([])
# plt.show()

# plt.imshow(roi_hist)
# plt.title('roi_hist'), plt.xticks([]), plt.yticks([])
# plt.show()

'''
while True:
    ret, frame = cap.read()     # read the rest of the video
    
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # arg1: original image
        # arg2: channel
        # arg3: roi histogram
        # arg4: ranges
        # arg5: scale
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # draw bounding box on the image
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
        cv2.imshow('img2', img2)
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break
    else:
        break
cv2.destroyAllWindows()
cap.release()
'''

# not good that window size is the same even when object moves closer/farther
# need to adapt window size with size and rotation of target
# CAMshift --> continuous adaptive meanshift
    # first apply meanshift
    # when that converges, update window size as s = 2 * sqrt(M00/256)
    # also calculate orientation of best fitting ellipse
    # applies meanshift again with the new scaled search window and previous location
    # continues until required accuracy is met

while True:
    ret, frame = cap.read()
    
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # draw it on original image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, 255, 2)
        cv2.imshow('img2', img2)
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break
    else:
        break
cv2.destroyAllWindows()
cap.release()