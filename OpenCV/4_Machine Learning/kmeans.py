# INPUTS
# 1. samples: type np.float32, and each feature in a single column
# 2. nclusters(K): number of clusters required at end
# 3. criteria: iteration termination criteria (type, max_iter, epsilon)
    # a. type: cv2.TERM_CRITERIA_EPS --> stops if accuracy epsilon is reached
    # cv2.TERM_CRITERIA_MAX_ITER --> stops after number of iterations
    # cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER --> any of the conditions
    # b. max_iter --> max number of iterations
    # c. epsilon --> required accuracy
# 4. attempts: number of times the algorithm is executed suing different initial labellings
# 5. flags: how initial centers are taken
    # cv2.KMEANS_PP_CENTERS and cv2.KMEANS_RANDOM_CENTERS

# OUTPUTS
# 1. compactness: sum of squared distance from each point to center
# 2. labels: label array where each element is marked 0, 1, ... depending on which cluster they belong to
# 3. centers: array of centers of cluters

import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
# EXAMPLE 1: SINGLE FEATURE
x = np.random.randint(25, 100, 25)
y = np.random.randint(175, 255, 25)
z = np.hstack((x, y))   # stack the two arrays into a single array
z = z.reshape((50, 1))
z = np.float32(z)
# plt.hist(z, 256, [0, 256])
# plt.show()

# define stopping criteria and flags
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

compactness, labels, centers = cv2.kmeans(z, 2, None, criteria, 10, flags)

# split data into different clusters depending on labels
A = z[labels == 0]
B = z[labels == 1]

plt.hist(A, 256, [0, 256], color = 'r')
plt.hist(B, 256, [0, 256], color = 'b')
plt.hist(centers, 64, [0, 256], color = 'y')
plt.show()
'''

'''
# EXAMPLE 2: MULTIPLE FEATURES
X = np.random.randint(25, 50, (25, 2))  # 25x2 numbers in range 25, 50
Y = np.random.randint(60, 85, (25, 2))
Z = np.vstack((X, Y))                   # vertical stack: row-wise
Z = np.float32(Z)

# define criteria and apply kmeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
compant, label, center = cv2.kmeans(Z, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# separate the data
# ravel(), aka flatten(), turns [[0], [1]] into [0, 1]
A = Z[label.ravel() == 0]
B = Z[label.ravel() == 1]

plt.scatter(A[:, 0], A[:, 1])
plt.scatter(B[:, 0], B[:, 1], color = 'r')
plt.scatter(center[:, 1], center[:, 1], s = 80, c = 'y', marker = 's')
plt.xlabel('Height'), plt.ylabel('Weight')
plt.show()
'''

# EXAMPLE 3: COLOUR QUANTIZATION
# process of reducing number of colours in an image
# 3 colours R, G, B, so reshape image to array of Mx3 size
    # M is the number of pixels
# after clustering, apply centroid values to all pixels
# then reshape back to shape of original image
img = cv2.imread('apple.jpg')
Z = img.reshape((-1, 3))
Z = np.float32(Z)

# define criteria and apply kmeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
compact, label, center = cv2.kmeans(Z, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert back to uint8 and rebuild original image
center = np.uint8(center)       # the 8 RGB centroids

# for each pixel, know what label it belongs to via label.flatten
# assign that pixel the RGB value of that label's centroid
    # ex. array(9, 8, 7, 6, 5) at index [2, 4] returns [7, 5]
res = center[label.flatten()]   
res2 = res.reshape((img.shape))

cv2.imshow('res2', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()