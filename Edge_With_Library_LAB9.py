from scipy import ndimage
import cv2
import numpy as np

scale = 1
delta = 0
ddepth = cv2.CV_16S

# Sobel
src = cv2.imread(f"/home/niit/Desktop/Archit/image.orig/100.jpg", cv2.IMREAD_COLOR)
src = cv2.GaussianBlur(src, (3, 3), 0)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

cv2.imshow("Sobel", grad)

# ---------------------------------------------------------
# Prewitt
img = cv2.imread('/home/niit/Desktop/Archit/image.orig/100.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)

kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
img_prewitt = img_prewittx + img_prewitty

cv2.imshow("Original Image", img)
cv2.imshow("Prewitt", img_prewitt)

# --------------------------------------------------------
# Robert
roberts_cross_v = np.array([[1, 0], [0, -1]])
roberts_cross_h = np.array([[0, 1], [-1, 0]])
img = cv2.imread("D:\\SEM-5\\Digital Image Processing\\Practical\\image.orig\\100.jpg", 0).astype('float64')
img /= 255.0
vertical = ndimage.convolve(img, roberts_cross_v)
horizontal = ndimage.convolve(img, roberts_cross_h)
edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))
cv2.imshow("Robert", edged_img)
edged_img *= 255

cv2.waitKey(0)
