# when stacking two images together, may not look good due to sudden discontinuity
# blending with pyramids is more seamless without large reduction in data

# 1. load the two images of apple and orange
# 2. find the gaussian pyramids for apple and orange (ex. 6 levels)
# 3. from gaussian pyramids, find their laplacian pyramids
# 4. join the left half of apple and right half of orange in each levels of laplacian pyramids
# 5. from joint image pyramids, reconstruct the original image

import cv2
import numpy as np

A = cv2.imread('apple.jpg')
B = cv2.imread('orange.jpg')

# generate Gaussian pyramid for A (6 layers)
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B (6 layers)
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    print(str(G.shape) + "\n")
    gpB.append(G)

# generate Laplacian pyramid for A
lpA = [gpA[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpA[i])
    print(str(GE.shape))
    print(str(gpA[i-1].shape) + "\n")
    L = cv2.subtract(gpA[i-1], GE)
    lpA.append(L)

# generate Laplacian pyramid for B
lpB = [gpB[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpB[i])
    #print(str(GE.size))
    #print(str(gpB[i-1].size))
    L = cv2.subtract(gpB[i-1], GE)
    lpB.append(L)

# add left and right halves of images in each level
LS = []
for la, lb in zip(lpA, lpB):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0:int(cols/2)], lb[:, int(cols/2):]))    # join a sequence of arrays along a new axis
    LS.append(ls)

# reconstruct the high-res image
ls_ = LS[0]
for i in range(1, 6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

# another image, with direct connecting each half
real = np.hstack((A[:, 0:int(cols/2)], B[:, int(cols/2):]))

cv2.imshow('pyramid blend', ls_)
cv2.imshow("directly blending", real)
cv2.waitKey(0)
cv2.destroyAllWindows()