import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt

for i in range(100, 105):
    img_path = f"D:\\SEM-5\\Digital Image Processing\\Practical\\image.orig\\{i}.jpg"
    img = cv2.imread(img_path)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = grayscale


    def dist(point1, point2):
        return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


    def butterworth_LP(D_0, imgShape, n):
        b = np.zeros(imgShape[:2])
        row, col = imgShape[:2]
        center = (row / 2, col / 2)
        for x in range(col):
            for y in range(row):
                b[y, x] = 1 / (1 + (dist((y, x), center) / D_0) ** (2 * n))
        return b


    def butterworth_HP(D_0, imgShape, n):
        b = np.zeros(imgShape[:2])
        row, col = imgShape[:2]
        center = (row / 2, col / 2)
        for x in range(col):
            for y in range(row):
                if dist((y, x), center) != 0:
                    b[y, x] = 1 / (1 + D_0 / (dist((y, x), center)) ** (2 * n))
        return b


    fourier_transform = np.fft.fft2(img)
    center_shift = np.fft.fftshift(fourier_transform)

    fourier_noisy = 20 * np.log(np.abs(center_shift))

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    for x in range(0, rows):
        for y in range(0, cols):
            if x + y == cols:
                for i in range(0, 10):
                    center_shift[x - i, y] = 1

    # Butterworth low pass filter on the grayscale image
    filtered = center_shift * butterworth_LP(80, img.shape, 3)

    f_shift = np.fft.ifftshift(center_shift)
    denoised_image = np.fft.ifft2(f_shift)
    denoised_image = np.real(denoised_image)

    f_ishift_blpf = np.fft.ifftshift(filtered)
    denoised_image_blpf = np.fft.ifft2(f_ishift_blpf)
    denoised_image_blpf = np.real(denoised_image_blpf)

    fourier_noisy_noise_removed = 20 * np.log(np.abs(center_shift))
    cv2.destroyAllWindows()

    # Butterworth high pass filter on the grayscale image
    filtered_h = center_shift * butterworth_HP(50, img.shape, 5)

    f_shift = np.fft.ifftshift(center_shift)
    denoised_image = np.fft.ifft2(f_shift)
    denoised_image = np.real(denoised_image)

    f_ishift_bhpf = np.fft.ifftshift(filtered_h)
    denoised_image_bhpf = np.fft.ifft2(f_ishift_bhpf)
    denoised_image_bhpf = np.real(denoised_image_bhpf)

    cv2.imshow("Original", img)
    cv2.imshow("Butterworth lpf", denoised_image_blpf)
    cv2.imshow("Butterworth hpf", denoised_image_bhpf)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # -----------------------------------------------------------------------------------------------------------------
    # Normal Low pass filter


    def lowPassFiltering(img, size):
        height, width = img.shape[0:2]
        h1, w1 = int(height / 2), int(width / 2)
        im2 = np.zeros((h, w), np.uint8)
        im2[h1 - int(size / 2):h1 + int(size / 2), w1 - int(size / 2):w1 + int(
            size / 2)] = 1
        im3 = im2 * img
        return im3


    h, w = grayscale.shape

    for i in range(3000):  # Add 3000 Noise Points
        x = np.random.randint(0, h)
        y = np.random.randint(0, w)
        grayscale[x, y] = 255

    # Fourier transform
    img_dft = np.fft.fft2(grayscale)
    dft_shift = np.fft.fftshift(img_dft)

    # Low-pass filter
    dft_shift = lowPassFiltering(dft_shift, 200)
    res = np.log(np.abs(dft_shift))

    # Inverse Fourier Transform
    idft_shift = np.fft.ifftshift(dft_shift)
    ifimg = np.fft.ifft2(idft_shift)
    ifimg = np.abs(ifimg)
    cv2.imshow("Filtered by LPF", np.int8(ifimg))

    # Draw pictures
    plt.subplot(131), plt.imshow(grayscale, 'gray'), plt.title('Original Image')
    plt.axis('off')
    plt.subplot(132), plt.imshow(res, 'gray'), plt.title('Low pass Filter')
    plt.axis('off')
    plt.subplot(133), plt.imshow(np.int8(ifimg), 'gray'), plt.title('filtered')
    plt.axis('off')
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()