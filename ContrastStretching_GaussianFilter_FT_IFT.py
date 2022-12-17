import cv2
import numpy as np
from matplotlib import pyplot as pl

img_path = f"D:\\SEM-5\\Digital Image Processing\\Practical\\image.orig\\100.jpg"
img = cv2.imread(img_path)
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------------------------------------------------------------------
# QN 1 - Histogram equalization and Contrast stretching with histogram plotting
# Histogram Equalization
hist_orig = cv2.calcHist([grayscale], [0], None, [256], [0, 256])
histg = cv2.equalizeHist(grayscale)
hist_equalized = cv2.calcHist([histg], [0], None, [256], [0, 256])

# Contrast Stretching
xp = [0, 64, 128, 192, 255]
fp = [0, 16, 128, 240, 255]
x = np.arange(256)
table = np.interp(x, xp, fp).astype('uint8')
contrast_image = cv2.LUT(grayscale, table)
hist_contrast = cv2.calcHist([contrast_image], [0], None, [256], [0, 256])

# Histogram Plotting
fig, (ax1, ax2, ax3) = pl.subplots(3, 1)
ax1.plot(hist_orig)
ax2.plot(hist_equalized)
ax3.plot(hist_contrast)
pl.tight_layout()
pl.show()
s = " " * 80
pl.plot(hist_contrast)
final = np.hstack((grayscale, histg, contrast_image))
cv2.imshow("Grayscale" + s + "Histogram Equalized" + s + "Contrast Strecthed", final)
cv2.waitKey(0)

# -----------------------------------------------------------------------------------------
# QN 2 - Gaussian Filter with varying sigma
for i in [3, 5, 7, 9, 11, 15]:
    gaussian_blur = cv2.GaussianBlur(src=img, ksize=(i, i), sigmaX=(i - 1) / 6, sigmaY=0)
    cv2.imshow(f"Gaussian Filter of {i}x{i}", gaussian_blur)
    cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------------------------------------------------------------------------
# QN 3 - Fourier and inverse fourier transform
dft = cv2.dft(np.float32(grayscale), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
mag, phase = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])
spec = np.log(mag) / 30
real, imag = cv2.polarToCart(mag, phase)
back = cv2.merge([real, imag])
back_ishift = np.fft.ifftshift(back)
img_back = cv2.idft(back_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
min, max = np.amin(img_back, (0, 1)), np.amax(img_back, (0, 1))
img_back = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow("ORIGINAL", grayscale)
cv2.imshow("SPECTRUM", spec)
cv2.imshow("ORIGINAL DFT/IFT ROUND TRIP", img_back)
cv2.waitKey(0)
