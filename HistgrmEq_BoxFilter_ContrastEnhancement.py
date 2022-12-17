import cv2
import matplotlib.pyplot as plt
import numpy as np

for i in range(110, 115):
    # Load the image
    img_path = f"D:\\SEM-5\\Digital Image Processing\\Practical\\image.orig\\{i}.jpg"
    img = cv2.imread(img_path)

    # Gray scale
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # QN1 Equalization
    hist_orig = cv2.equalizeHist(grayscale)
    cv2.imshow("Equalized", hist_orig)
    hist_orig = cv2.calcHist([grayscale], [0], None, [256], [0, 256])
    plt.plot(hist_orig)
    equ = cv2.equalizeHist(grayscale)
    hist_new = cv2.calcHist([equ], [0], None, [256], [0, 256])
    plt.plot(hist_new)
    plt.show()

    # QN 2-3 Convolution and Enhancement
    xp = [0, 64, 128, 192, 255]
    fp = [0, 64, 128, 192, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')

    arr = [3, 5, 7, 9, 11, 15]
    for i in arr:
        kern = np.ones((i, i), np.float32) / i ** 2
        con = cv2.filter2D(src=grayscale, ddepth=-1, kernel=kern)
        hist = cv2.equalizeHist(con)
        contrast_image = cv2.LUT(con, table)
        final = np.hstack((con, hist, contrast_image))
        space = 70 * " "
        cv2.imshow(
            f"Convoluted using {i}x{i}box filter" + space + "Histogram Enhancement" + space + "Contrast Enhancement",
            final)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

cv2.waitKey(0)