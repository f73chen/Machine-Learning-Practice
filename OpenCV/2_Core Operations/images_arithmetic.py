import cv2
import numpy as np

'''
# add tow images of the same depth and type, or second image is a scalar value
x = np.uint8([250])
y = np.uint8([10])

print(cv2.add(x, y))    # 260 --> 255
print(x + y)            # 260 % 256 = 4

# blend images by adding with different weights
# g(x) = (1 - a) * f0(x) + a * f1(x), 0 <= a <= 1
img1 = cv2.imread(cv2.samples.findFile("starry_night.jpg"), -1)
img2 = cv2.flip(cv2.imread(cv2.samples.findFile("starry_night.jpg"), -1), 0)

# dst = a * img1 + b * img2 + g
dst = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

img1 = cv2.flip(cv2.imread(cv2.samples.findFile("starry_night.jpg"), -1), -1)
img2 = cv2.imread('OpenCV.png', -1)

# want to put OpenCV logo on top-left corner, so create ROI
rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]      # select top left of starry_night

# create mask of logo and its inverse mask
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)   
# in mask, pixels between 20 and 255 brightness becomes 255, rest becomes 0
ret, mask = cv2.threshold(img2gray, 20, 255, cv2.THRESH_BINARY)
# in inverse, pixels between 20 and 255 brightness becomes 0, rest becomes 255
mask_inv = cv2.bitwise_not(mask)
cv2.imshow('res22', mask_inv)

# black out the area of the logo in ROI (logo area becomes 0)
img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)

# take only region of logo from logo image (non-logo area becomes 0)
img2_fg = cv2.bitwise_and(img2, img2, mask = mask)

# merge starry_night background with logo foreground
dst = cv2.add(img1_bg, img2_fg)

# replace original starry_night ROI with modified pixels
img1[0:rows, 0:cols] = dst

cv2.imshow('res', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()