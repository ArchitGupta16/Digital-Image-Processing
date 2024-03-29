import os
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian, convolve2d
import matplotlib.pyplot as plt


def blur(img, kernel_size=3):
    dummy = np.copy(img)
    h = np.eye(kernel_size) / kernel_size
    dummy = convolve2d(dummy, h, mode='valid')
    return dummy


def add_gaussian_noise(img, sigma):
    gauss = np.random.normal(0, sigma, np.shape(img))
    noisy_img = img + gauss
    noisy_img[noisy_img < 0] = 0
    noisy_img[noisy_img > 255] = 255
    return noisy_img


def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s=img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return dummy


def gaussian_kernel(kernel_size=3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# Load image and convert it to gray scale
file_name = os.path.join('D:\\SEM-5\\Digital Image Processing\\Practical\\image.orig\\100.jpg')
img = rgb2gray(plt.imread(file_name))

# Blur the image
blurred_img = blur(img, kernel_size=7)

# Add Gaussian noise
noisy_img = add_gaussian_noise(blurred_img, sigma=20)

# Apply Wiener Filter
kernel = gaussian_kernel(5)
filtered_img = wiener_filter(noisy_img, kernel, K=10)

# Display results
display = [img,blurred_img,noisy_img, filtered_img]
label = ['Original Image', 'Blurred Image', 'Noisy Image', 'Wiener Filter applied']

fig = plt.figure(figsize=(12, 10))

for i in range(len(display)):
    fig.add_subplot(2, 2, i + 1)
    plt.imshow(display[i], cmap='gray')
    plt.title(label[i])

plt.show()

import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("D:\\SEM-5\\Digital Image Processing\\Practical\\image.orig\\100.jpg")

hist1 = cv2.calcHist([image], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([image], [1], None, [256], [0, 256])
hist3 = cv2.calcHist([image], [2], None, [256], [0, 256])

plt.subplot(221), plt.imshow(image)
plt.subplot(222), plt.plot(hist1), plt.plot(hist2), plt.plot(hist3)
plt.xlim([0, 256])

plt.show()
