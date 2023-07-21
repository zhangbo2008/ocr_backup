# 得到图片的边缘.
import cv2,math
import numpy as np


import cv2 as cv
path='tt.jpg'
image=cv.imread(path)
# image=cv.cvtColor(image,cv.COLOR_RGB2GRAY)



def calculateElement(img):
    # 根据图片大小粗略计算腐蚀 或膨胀所需核的大小
    sp = img.shape
    width = sp[1]  # width(colums) of image
    kenaly = math.ceil((width / 400.0) * 12)
    kenalx = math.ceil((kenaly / 5.0) * 4)
    a = (int(kenalx), int(kenaly))

    return a








img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图片

img_blurred = cv2.filter2D(img_gray, -1,kernel=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32))  #对图像进行滤波,是锐化操作
img_blurred = cv2.filter2D(img_blurred, -1, kernel=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32))
cv2.imwrite('img_blurred.jpg',img_blurred)










threshold = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]



element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
element2 = cv2.getStructuringElement(cv2.MORPH_RECT, calculateElement(threshold))


dilation = cv2.dilate(threshold, element1, iterations=1)
binary = cv2.erode(dilation, element1, iterations=1)

# 文字膨胀与腐蚀使其连成一个整体
erosion = cv2.erode(binary, element2, iterations=1)
dilation = cv2.dilate(erosion, element1, iterations=1)









canny = cv2.Canny(threshold, 100, 150)
# show(canny, "canny")
kernel = np.ones((3, 3), np.uint8)
dilate = cv2.dilate(canny, kernel, iterations=5)
# show(dilate, "dilate")








# cv.imshow('THRESH_BINRY',binary)
cv2.imwrite('binary.jpg',dilate)




if 0:
    img=cv2.imread('tt.jpg')
    """
    将图片灰度化，并锐化滤波
    :param img:  输入RGB图片
    :param image_name:  输入图片名称，测试时使用
    :param save_path:   滤波结果保存路径，测试时使用
    :return: 灰度化、滤波后图片
    """
 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图片
 
    img_blurred = cv2.filter2D(img_gray, -1,kernel=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32))  #对图像进行滤波,是锐化操作
    img_blurred = cv2.filter2D(img_blurred, -1, kernel=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32))
    cv2.imwrite('img_blurred.jpg',img_blurred)

    """
    求取梯度，二值化
    :param img_blurred: 滤波后的图片
    :param image_name: 图片名，测试用
    :param save_path: 保存路径，测试用
    :return:  二值化后的图片
    """
    gradX = cv2.Sobel(img_blurred, ddepth=cv2.CV_32F, dx=1, dy=0) # sobel算子,计算梯度, 也可以用canny算子替代
    gradY = cv2.Sobel(img_blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
    img_gradient = cv2.subtract(gradX, gradY) #使用减法作图像融合？
    #img_gradient = cv2.addWeighted(gradX,2, gradY,2,0)
    img_gradient = cv2.convertScaleAbs(img_gradient) #用convertScaleAbs()函数将其转回原来的uint8形式
 
    # 这里改进成自适应阈值,貌似没用
    img_binary = cv2.adaptiveThreshold(img_gradient, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -3)
    # cv2.imshow("img_binary", img_binary)
 
    # 这里调整了kernel大小(减小),腐蚀膨胀次数后(增大),出错的概率大幅减小
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_closed = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel) #形态学关操作
    mg_closed = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel)  #形态学开操作
    img_closed = cv2.erode(mg_closed, None, iterations=9)    #腐蚀
    img_closed = cv2.dilate(img_closed, None, iterations=9)  # 膨胀
 
    cv2.imwrite('img_closed.jpg',img_closed)

    
    def point_judge(center, bbox):
        """
        用于将矩形框的边界按顺序排列
        :param center: 矩形中心的坐标[x, y]
        :param bbox: 矩形顶点坐标[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        :return: 矩形顶点坐标,依次是 左下, 右下, 左上, 右上
        """
        left = []
        right = []
        for i in range(4):
            if bbox[i][0] > center[0]:  # 只要是x坐标比中心点坐标大,一定是右边
                right.append(bbox[i])
            else:
                left.append(bbox[i])
        if right[0][1] > right[1][1]:  # 如果y点坐标大,则是右上
            right_down = right[1]
            right_up = right[0]
        else:
            right_down = right[0]
            right_up = right[1]
    
        if left[0][1] > left[1][1]:  # 如果y点坐标大,则是左上
            left_down = left[1]
            left_up = left[0]
        else:
            left_down = left[0]
            left_up = left[1]
        return left_down, right_down, left_up, right_up
    
    

    """
    根据二值化结果判定并裁剪出身份证正反面区域
    :param img: 原始RGB图片
    :param img_closed: 二值化后的图片
    :return: 身份证正反面区域
    """
    (contours, _) = cv2.findContours(img_closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 求出框的个数
    # 这里opencv如果版本不对（4.0或以上）会报错，只需把(contours, _)改成 (_, contours, _)
 
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 按照面积大小排序
    print(1)
    countours_res = []
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])  # 计算面积

        if (area <= 0.999 * img.shape[0] * img.shape[1]) and (area >= 0.05 * img.shape[0] * img.shape[1]):
            # 人为设定,身份证正反面框的大小不会超过整张图片大小的0.4,不会小于0.05(这个参数随便设置的)
            rect = cv2.minAreaRect(contours[i])  # 最小外接矩,返回值有中心点坐标,矩形宽高,倾斜角度三个参数
            box = cv2.boxPoints(rect)  #将rect使用boxPoints进行提取矩形的4个角点
            left_down, right_down, left_up, right_up = point_judge([int(rect[0][0]), int(rect[0][1])], box)
            src = np.float32([left_down, right_down, left_up, right_up])  # 这里注意必须对应
 
            dst = np.float32([[0, 0], [int(max(rect[1][0], rect[1][1])), 0], [0, int(min(rect[1][0], rect[1][1]))],
                              [int(max(rect[1][0], rect[1][1])),
                               int(min(rect[1][0], rect[1][1]))]])  # rect中的宽高不清楚是个怎么机制,但是对于身份证,肯定是宽大于高,因此加个判定
            m = cv2.getPerspectiveTransform(src, dst)  # 得到投影变换矩阵
            result = cv2.warpPerspective(img, m, (int(max(rect[1][0], rect[1][1])), int(min(rect[1][0], rect[1][1]))),
                                         flags=cv2.INTER_CUBIC)  # 投影变换
            countours_res.append(result)
    countours_res  # 返回身份证区域
print(15)