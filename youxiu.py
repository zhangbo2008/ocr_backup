import cv2
import numpy as np
# from cnocr import CnOcr

def show(image, window_name):
    # cv2.namedWindow(window_name, 0)
    cv2.imwrite(window_name+'.png', image)


image = cv2.imread('t1.png')
import cv2  #######边缘检测.
import argparse
import imutils
import cv2

import numpy as np
kernel = np.ones((1, 5), np.uint8)
# img = cv2.imread('tmp99.png')
# img = cv2.imread('data/tq.jpg')
img = image
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE) 
cv2.imwrite("img2.png", binary)   
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=2) # 二值化.
contours = cv2.findContours(binary,cv2.RETR_CCOMP  ,cv2.CHAIN_APPROX_SIMPLE)  # 参数说明;https://docs.opencv.org/4.0.0/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71 
contours = imutils.grab_contours(contours) #适配cv2各个版本.
contours = sorted(contours, key = cv2.contourArea, reverse = True)[0]
# binary=cv2.drawContours(img,contours,-1,(0,255,255),1)  
# cv2.imwrite("img.png", binary)




epsilon = 0.02 * cv2.arcLength(contours, True)
approx = cv2.approxPolyDP(contours, epsilon, True)
n = []
for x, y in zip(approx[:, 0, 0], approx[:, 0, 1]):
    n.append((x, y))
n = sorted(n)
sort_point = []
n_point1 = n[:2]
n_point1.sort(key=lambda x: x[1])
sort_point.extend(n_point1)
n_point2 = n[2:4]
n_point2.sort(key=lambda x: x[1])
n_point2.reverse()
sort_point.extend(n_point2)
p1 = np.array(sort_point, dtype=np.float32)
h = sort_point[1][1] - sort_point[0][1]
w = sort_point[2][0] - sort_point[1][0]
pts2 = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)

M = cv2.getPerspectiveTransform(p1, pts2)
dst = cv2.warpPerspective(image, M, (w, h))
# print(dst.shape)
show(dst, "dst2")
if w < h:
    dst = np.rot90(dst)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	