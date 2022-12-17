import numpy as np
import cv2
from matplotlib import pyplot as plt

img_path = "1.jpg"
img = cv2.imread(img_path)
cv2.imshow("Noisy Image", img)

# Denoising

# 1.Gaussian

img1 = cv2.GaussianBlur(src=img, ksize=(7, 7), sigmaX=2 / 3, sigmaY=0)
cv2.imshow(f"Denoised Image using gaussian filter", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --------------------------------------------------------
# 2.Arithmetic Mean

img2 = cv2.blur(img, (7, 7))
cv2.imshow("Arithmetic Mean Filter", img2)

# -------------------------------------------------------------------
# 3.Geometric mean Filter

img_gray = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE).astype(float)
rows, cols = img_gray.shape[:2]
ksize = 5
padsize = int((ksize - 1) / 2)
pad_img = cv2.copyMakeBorder(img_gray, *[padsize] * 4, cv2.BORDER_DEFAULT)
geomean1 = np.zeros_like(img_gray)
for r in range(rows):
    for c in range(cols):
        geomean1[r, c] = np.prod(pad_img[r:r + ksize, c:c + ksize]) ** (1 / (ksize ** 2))
geomean1 = np.uint8(geomean1)
cv2.imshow('Geometric mean', geomean1)

# --------------------------------------------------------
# 4.Contraharmonic Mean Filter

img = cv2.imread("1.jpg")


def contraharmonic_mean(image, size, Q):
    num = np.power(image, Q + 1)
    denom = np.power(image, Q)
    kernel = np.full(size, 1.0)
    result = cv2.filter2D(num, -1, kernel) / cv2.filter2D(denom, -1, kernel)
    return result


ans = contraharmonic_mean(img, 3, 2)
cv2.imshow("ContraHarmonic", ans)
cv2.waitKey(0)


# --------------------------------------------
# Fourier

# ------------------------------------------------------
# Notch Filter

img = cv2.imread("1.jpg",0)


def notch_reject_filter(shape, d0=9, u_k=0, v_k=0):
    P, Q = shape
    # Initialize filter with zeros
    H = np.zeros((P, Q))

    # Traverse through filter
    for u in range(0, P):
        for v in range(0, Q):
            # Get euclidean distance from point D(u,v) to the center
            D_uv = np.sqrt((u - P / 2 + u_k) ** 2 + (v - Q / 2 + v_k) ** 2)
            D_muv = np.sqrt((u - P / 2 - u_k) ** 2 + (v - Q / 2 - v_k) ** 2)

            if D_uv <= d0 or D_muv <= d0:
                H[u, v] = 0.0
            else:
                H[u, v] = 1.0

    return H


f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
phase_spectrumR = np.angle(fshift)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

img_shape = img.shape

H1 = notch_reject_filter(img_shape, 4, 38, 30)
H2 = notch_reject_filter(img_shape, 4, -42, 27)
H3 = notch_reject_filter(img_shape, 2, 80, 30)
H4 = notch_reject_filter(img_shape, 2, -82, 28)

NotchFilter = H1 * H2 * H3 * H4
NotchRejectCenter = fshift * NotchFilter
NotchReject = np.fft.ifftshift(NotchRejectCenter)
inverse_NotchReject = np.fft.ifft2(NotchReject)  # Compute the inverse DFT of the result

Result = np.abs(inverse_NotchReject)
plt.subplot(222)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(221)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('magnitude spectrum')

plt.subplot(223)
plt.imshow(magnitude_spectrum * NotchFilter, "gray")
plt.title("Notch Reject Filter")

plt.subplot(224)
plt.imshow(Result, "gray")
plt.title("Result")

plt.show()

# Bandpass Filter
img = cv2.imread("1.jpg")
imgg = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
dft = cv2.dft(np.float32(imgg), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

rows, cols = imgg.shape
crow, ccol = int(rows / 2), int(cols / 2)
mask = np.zeros((rows, cols, 2), np.uint8)
r_out = 80
r_in = 10
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                           ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
mask[mask_area] = 1

fshift = dft_shift * mask

fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]),
                                where=cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) > 0)

f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
cv2.imshow("Filtered using bandpass", img_back)
cv2.waitKey(0)
