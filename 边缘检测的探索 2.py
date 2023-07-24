
#======从mask 变到 图像的四角定位.

import cv2  #######边缘检测.
import argparse
import imutils
import cv2

import numpy as np




import argparse
import imutils
import cv2

import cv2
import numpy as np
 
#写一个方法用于展示图像，展示后按任意键继续下行代码
def cv_show(title,img):
    cv2.imshow(title,img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    return
 
img = cv2.imread('debug999.png',0).astype(np.uint8)*1 

cv2.imwrite('img.png',img)

#就是上一章的内容，具体就是会输出一个轮廓图像并返回一个轮廓数据
def draw_contour(img,color,width):
    kernel = np.ones((1, 5), np.uint8)
    if len(img.shape)>2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#转换灰度图
    else:
        gray=img
    cv2.imwrite('gray.png',gray)
    ret, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=2) # 二值化. 
    cv2.imwrite('binary.png',binary)
    contours,hierarchy = cv2.findContours(binary,cv2.RETR_CCOMP  ,cv2.CHAIN_APPROX_SIMPLE)



    contours = sorted(contours, key = cv2.contourArea, reverse = True)[0]
    copy = img.copy()
    resoult = cv2.drawContours(copy,contours[:],-1,color,width)
    cv2.imwrite('resoult.png',resoult)
    return contours
 
 
contour = draw_contour(img,(0,0,255),2)
print(type(contour))    

epsilon = 0.02 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)
print(type(approx),approx.shape)

res = cv2.drawContours(img.copy(),[approx],-1,(0,255,255),2)


ttt=cv2.imread('data/tq2.png')
ttt=cv2.resize(ttt,(1280,845))
res2=cv2.drawContours(ttt.copy(),[approx],-1,(0,255,255),2)
cv2.imwrite('res.png',res) #打印最后的多边形矿.
cv2.imwrite('res2.png',res2) #打印最后的多边形矿.

print('对应的原始图片是','data/tq2.png')


