import cv2
import numpy as np

img = cv2.imread('chess2.jpg')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)

# find centroids for sub-pixel accuracy
# input: image --> the 8-bit single-channel image to be labeled
# opt input1: connectivity --> 8 for 8-way, 4 for 4-way
# opt input2: ltype --> output image label type (CV_32S or CV_16U)
# opt input3: ccltype --> connected components algorithm type
    # cv.CCL_WU --> SAUF for 8-way, SAUF for 4-way
    # cv.CCL_DEFAULT --> BBDT for 8-way, SAUF for 4-way
    # cv.CCL_GRANA --> BBDT for 8-way, SAUF for 4-way
# output1: labels --> destination labeled image
# output2: stats --> select with (label, COLUMN), where COLUMN is:
    # CC_STAT_LEFT --> the leftmost x coordinate which is the inclusive start of the bounding box in the horizontal direction
    # CC_STAT_TOP --> the topmost y coordinate which is the inclusive start of the bounding box in the vertical direction
    # CC_STAT_WIDTH --> the horizontal size of the bounding box
    # CC_STAT_HEIGHT --> the vertical size of the bounding box
    # CC_STAT_AREA --> total area (in pixels) of the connected component
# output3: centroids --> centroid for each label, including the background label
    # accessed via centroids(label, 0) for x and centroids(label, 1) for y
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

ordered = np.unique(labels)[1:]
ordered = sorted(ordered, key = lambda x: stats[x, 4])
for label in ordered:
    print(str(label) + ", area: " + str(stats[label, 4]))
    mask = np.zeros(labels.shape)
    # print(mask.shape)
    mask[labels == label] = 255
    cv2.imshow('label', mask)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

