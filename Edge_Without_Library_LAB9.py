import cv2
import numpy as np
from scipy import ndimage

# Robert Without Library
roberts_cross_v = np.array([[1, 0],
                            [0, -1]])

roberts_cross_h = np.array([[0, 1],
                            [-1, 0]])

img = cv2.imread("D:\\SEM-5\\Digital Image Processing\\Practical\\image.orig\\100.jpg", 0).astype('float64')
img /= 255.0
vertical = ndimage.convolve(img, roberts_cross_v)
horizontal = ndimage.convolve(img, roberts_cross_h)

edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))
edged_img *= 255

cv2.imwrite("Robert.jpg", edged_img)

# ------------------------------------------------------------------------------
# Prewitt without Library
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
img_prewittx = cv2.filter2D(img, -1, kernelx)
img_prewitty = cv2.filter2D(img, -1, kernely)
img_prewitt = img_prewittx + img_prewitty

cv2.imshow("Prewitt", img_prewitt)

# --------------------------------------------------------------------------------
# Sobel without Library
img = cv2.imread("D:\\SEM-5\\Digital Image Processing\\Practical\\image.orig\\100.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
sobelx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sobely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
img_sobelx = cv2.filter2D(gray, -1, sobelx)
img_sobely = cv2.filter2D(gray, -1, sobely)
img_sobel = img_sobelx + img_sobely
cv2.imshow("Sobel", img_sobel)

cv2.waitKey(0)
