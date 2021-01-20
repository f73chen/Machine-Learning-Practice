from skimage.io import imread, imshow
from skimage.transform import rotate
import matplotlib.pyplot as plt

img = imread('tilted.jpg')
img_rot = rotate(img, angle = 45, resize = True)
imshow(img_rot)
plt.show()