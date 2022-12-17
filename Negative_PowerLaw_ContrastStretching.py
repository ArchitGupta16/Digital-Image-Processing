import cv2
import numpy as np
for i in range(110,116):
    #Load the image
    img_path = f"/home/niit/Desktop/Archit/image.orig/{i}.jpg"
    img = cv2.imread(img_path)

    # gray scale
    grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale",grayscale)

    # image negative
    gray_negative = abs(255-grayscale)
    cv2.imshow("Negative",gray_negative)

    # power law transformation
    gamma = 1.5
    gamma_img = np.array(255*(img/255)**gamma,dtype='uint8')
    cv2.imshow(f"Gamma={gamma}",gamma_img)

    # Contrast stretching
    xp = [0,64,128,192,255]
    fp = [0,64,128,192,255]
    x = np.arange(256)
    table = np.interp(x,xp,fp).astype('uint8')
    contrast_image = cv2.LUT(img,table)
    space = 89*" "
    cv2.imshow("Contrast Stretching",contrast_image)
    cv2.waitKey(4000)
    cv2.destroyAllWindows()