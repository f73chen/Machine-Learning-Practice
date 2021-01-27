# static camera takes video of background w/ objects moving in foreground
# need to extract moving foreground from static background
# harder with moving shadow

'''
# 1. BackgroundSubtractorMOG
# models each background pixel by a mixture of K gaussian distributions
# weights of mixture rep. time proportions that those colours stay in the scene
# probably background colours are the ones which stay longest
# Note: not available in python3?
'''

# 2. BackgroundSubtractorMOG2
# similar to above, but selects appropriate number of gaussian distributions for each pixel
# better adaptibility to varying scenes due to illumination changes etc.
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2() # detectShadows = True/False

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    
    cv2.imshow('frame', fgmask)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

'''
# 3. BackgroundSubtractorGMG
# combines statistical background image estimation and per-pixel Bayesian segmentation
# uses first few (default 120) frames for background modelling
# probabilistic foreground segmentation algo taht identifies possible fg objects w/ Bayesian inference
# newer observations more heavily weighted than old obs to accomodate var illumination
# opening & closing to remove noise
# Note: also not available in current version
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fgbg = cv2.createBackgroundSubtractorGMG()

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('frame', fgmask)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''