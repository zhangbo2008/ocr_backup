
#======从mask 变到 图像的四角定位.

import cv2  #######边缘检测.
import argparse
import imutils
import cv2

import numpy as np






import cv2
import numpy as np
 
#写一个方法用于展示图像，展示后按任意键继续下行代码
def cv_show(title,img):
    cv2.imshow(title,img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    return
 
img = cv2.imread('random.png')
img= np.array([[False,False,False,False]
,[False,True,True,False]
,[False,True,True,False]
,[False,False,False,False]

]).astype(np.uint8)*255             #######输入自定义mask




#就是上一章的内容，具体就是会输出一个轮廓图像并返回一个轮廓数据
def draw_contour(img,color,width):
    if len(img.shape)>2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#转换灰度图
    else:
        gray=img
    ret , binary = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)#转换成二值图
    contour,hierarchy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#寻找轮廓，都储存在contour中
    copy = img.copy()
    resoult = cv2.drawContours(copy,contour[:],-1,color,width)
    # cv2.imwrite('resoult.png',resoult)
    return contour
 
 
contour = draw_contour(img,(0,0,255),2)
print(type(contour))    
epsilon = 0.01*cv2.arcLength(contour[0],True)
approx = cv2.approxPolyDP(contour[0], epsilon, True)
print(type(approx),approx.shape)
 
res = cv2.drawContours(img.copy(),[approx],-1,(0,255,255),2)
cv2.imwrite('res.png',res) #打印最后的多边形矿.





