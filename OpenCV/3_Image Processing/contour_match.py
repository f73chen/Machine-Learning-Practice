import cv2
import numpy as np

img1 = cv2.imread("circle1.png", 0)
img2 = cv2.imread("circle2.png", 0)
img3 = cv2.imread("circle3.png", 0)
ret, thresh1 = cv2.threshold(img1, 50, 255, 0)
ret, thresh2 = cv2.threshold(img2, 50, 255, 0)
ret, thresh3 = cv2.threshold(img3, 50, 255, 0)

contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt1 = contours[0]
contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt2 = contours[0]
contours, hierarchy = cv2.findContours(thresh3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt3 = contours[0]

# match similarity between two shapes or contours
    # lower score = better match
# based on hu-moments: 7 moments invariant to translation, rotation, and scale
    # 7th is skew-invariant
ret1 = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
ret2 = cv2.matchShapes(cnt1, cnt3, 1, 0.0)
print("cnt1 vs cnt2: " + str(ret1))
print("cnt1 vs cnt3: " + str(ret2))

# note: remember to normalize images:
    # same background colour
    # closely cropped
    # similar image size
    # thresholding successfully segments the shape
# after the above, produced good results
cv2.imshow("img1", thresh1)
cv2.imshow("img2", thresh2)
cv2.imshow("img3", thresh3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# can compare images to known contours of letters/characters for OCR