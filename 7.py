# 得到图片的边缘.
import cv2
import numpy as np


import cv2 as cv
path='tt.jpg'
image=cv.imread(path)
image=cv.cvtColor(image,cv.COLOR_RGB2GRAY)


if 1: # 两种去噪方式. 腐蚀和膨胀!!!!!!!!
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    img_closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel) #形态学关操作
    image = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel)  #形态学开操作
    # img_closed = cv2.erode(mg_closed, None, iterations=9)    #腐蚀
    # img_closed = cv2.dilate(img_closed, None, iterations=9)  # 膨胀










threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]











canny = cv2.Canny(threshold, 100, 150)
# show(canny, "canny")
kernel = np.ones((3, 3), np.uint8)
dilate = cv2.dilate(canny, kernel, iterations=5)
# show(dilate, "dilate")




# cv.imshow('THRESH_BINRY',binary)
cv2.imwrite('binary7.jpg',dilate)



