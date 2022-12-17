import cv2


def non_overlapping():
    img = cv2.imread("/home/niit/Desktop/image.orig/1.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale image", gray)
    r = 3
    c = 3
    for i in range(0, img.shape[0] - r, r):
        for j in range(0, img.shape[0] - c, c):
            window = img[i:i + r, j:j + c]
            cv2.imshow("window", window)
            cv2.waitKey(10)
            cv2.destroyAllWindows()


def overlapping():
    img = cv2.imread("/home/niit/Desktop/image.orig/1.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale image", gray)
    r = 3
    c = 3
    for i in range(0, img.shape[0] - r):
        for j in range(0, img.shape[0] - c):
            window = img[i:i + r, j:j + c]
            cv2.imshow("window", window)
            cv2.waitKey(10)
            cv2.destroyAllWindows()


overlapping()
non_overlapping()
