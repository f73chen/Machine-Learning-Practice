# user interactively selects subject bounding box and some foreground & background
# algo uses mincut to separate nodes into source (fg) and sink (bg)

# img: input image
# mask: specify which areas are bg, fg, or probable bg/fg
    # cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD
    # or pass 0, 1, 2, 3 to image
# rect: coordinates of rectangle which includes the foreground object
    # x, y, w, h
# bgdModel, fgdModel: arrays used internally
    # create two np.float64 type 0 arrays of size (1, 65)
# iterCount: number of iterations of algorithm
# mode: whether drawing rectangle or final touchup strokes
    # cv2.GC_INIT_WITH_RECT, cv2.GC_INIT_WITH_MASK

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('messi5.jpg')
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (50, 50, 450, 290)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# grabcut modifies the mask, so pixels will be marked from 0-3 acc. to above
# modify so 0 & 2 set to 0 (background) and 1 & 3 set to 1 (foreground)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img*mask2[:, :, np.newaxis]

# manually label a new mask for clearer labelling
newmask = cv2.imread("touchups.png", 0)

# white = sure foreground, change mask = 1
# black = sure background, change mask = 0
mask[newmask < 20] = 0
mask[newmask > 235] = 1

mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img*mask[:, :, np.newaxis]

plt.imshow(img), plt.colorbar()
plt.show()