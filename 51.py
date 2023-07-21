import cv2  #######边缘检测.
import argparse
import imutils
import cv2

import numpy as np
kernel = np.ones((1, 5), np.uint8)
# img = cv2.imread('tmp99.png')
# img = cv2.imread('data/tq.jpg')
img = cv2.imread('data/tq2.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE) 
cv2.imwrite("img2.png", binary)   
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=25) # 二值化.
contours = cv2.findContours(binary,cv2.RETR_CCOMP  ,cv2.CHAIN_APPROX_SIMPLE)  # 参数说明;https://docs.opencv.org/4.0.0/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71 
contours = imutils.grab_contours(contours) #适配cv2各个版本.
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:2]
binary=cv2.drawContours(img,contours,-1,(0,255,255),3)  
cv2.imwrite("img.png", binary)  
