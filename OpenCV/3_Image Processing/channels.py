import cv2
import matplotlib.pyplot as plt

colour = cv2.imread('original_t.png', -1)
b, g, r = cv2.split(colour)
gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)

plt.subplot(221), plt.imshow(gray, cmap='Greys')
plt.title('gray'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(r, cmap='Greys')
plt.title('red'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(g, cmap='Greys')
plt.title('green'), plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.imshow(b, cmap='Greys')
plt.title('blue'), plt.xticks([]), plt.yticks([])

plt.show()