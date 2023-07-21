# 直线检测!


# 实际去测 效果.


# coding=utf-8
# 导入相应的python包
import cv2 
import numpy as np 
if 0:
    # 读取输入图片
    img = cv2.imread('tt.jpg') 
    # 将彩色图片灰度化
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    # 使用Canny边缘检测 
    edges = cv2.Canny(gray,50,200,apertureSize = 3) 
    # 进行Hough_line直线检测
    lines = cv2.HoughLines(edges,1,np.pi/180, 200) 
    print(lines)
    # 遍历每一个r和theta
    for i in range(len(lines)):
        r,theta = lines[i, 0, 0], lines[i, 0, 1]
        # 存储cos(theta)的值
        a = np.cos(theta)
        # 存储sin(theta)的值
        b = np.sin(theta) 
        # 存储rcos(theta)的值
        x0 = a*r 
        # 存储rsin(theta)的值 
        y0 = b*r  
        # 存储(rcos(theta)-1000sin(theta))的值
        x1 = int(x0 + 1000*(-b)) 
        # 存储(rsin(theta)+1000cos(theta))的值
        y1 = int(y0 + 1000*(a)) 
        # 存储(rcos(theta)+1000sin(theta))的值
        x2 = int(x0 - 1000*(-b)) 
        # 存储(rsin(theta)-1000cos(theta))的值
        y2 = int(y0 - 1000*(a))  
        # 绘制直线结果  
        cv2.line(img,(x1,y1), (x2,y2), (0,255,0),2) 
    # 保存结果
    cv2.imwrite('demo3.png', img) 
    # cv2.imshow("result", img)
    cv2.waitKey(0)




    # coding=utf-8
    # 导入相应的python包
    import cv2 
    import numpy as np 
    
    # 读取输入图片
    img = cv2.imread('tt.jpg') 
    # 将彩色图片灰度化
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    # 使用Canny边缘检测 
    edges = cv2.Canny(gray,100,200,apertureSize = 3) 
    # edges = cv2.cannyl(gray,100,200,apertureSize = 3) 
    # 进行Hough_line直线检测
    cv2.imwrite('Canny.jpg', edges) 
    cv2.imwrite('gray.jpg', gray) 
    lines = cv2.HoughLinesP(edges,1,np.pi/180, 80, 30, 10) 

    # 遍历每一条直线
    for i in range(len(lines)): 
        cv2.line(img,(lines[i, 0, 0],lines[i, 0, 1]), (lines[i, 0, 2],lines[i, 0, 3]), (0,255,0),2) 
    # 保存结果
    cv2.imwrite('HoughLinesP.jpg', img) 
    # cv2.imshow("result", img)
    cv2.waitKey(0)










    # coding=utf-8
    import cv2
    import numpy as np

    # 读取输入图片
    img0 = cv2.imread("tt.jpg")
    # 将彩色图片转换为灰度图片
    img = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)

    # 创建一个LSD对象
    lsd = cv2.createLineSegmentDetector(0)
    # 执行检测结果
    dlines = lsd.detect(img)
    # 绘制检测结果
    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        cv2.line(img0, (x0, y0), (x1,y1), (0,255,0), 1, cv2.LINE_AA)

    # 显示并保存结果
    cv2.imwrite('test3_r.jpg', img0)
    cv2.imshow("LSD", img0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





#==========参考:https://blog.csdn.net/wzw12315/article/details/105215218











#=========通过这个把图片