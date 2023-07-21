import cv2,math

import os, cv2
import sys, numpy as np
import math
import include.binaryzation as bz
import include.functions as func
import copy
import fileutil
import include.binaryzation as bz
img=cv2.imread('tt.jpg')
def calculateElement(img):
    # 根据图片大小粗略计算腐蚀 或膨胀所需核的大小
    sp = img.shape
    width = sp[1]  # width(colums) of image
    kenaly = math.ceil((width / 400.0) * 12)
    kenalx = math.ceil((kenaly / 5.0) * 4)
    a = (int(kenalx), int(kenaly))

    return a
def preprocess(gray, algoFunc):
    # 1. Sobel算子，x方向求梯度
    # sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)

    # 获取二值化阈值
    thr = bz.myThreshold()
    # threshold = thr.get1DMaxEntropyThreshold(gray)
    threshold = getattr(thr, algoFunc)(gray)
    if threshold <= 0:
        raise Exception("获取二值化阈值失败")

    # 2. 二值化
    ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # 获取核大小
    calculateElement(gray)

    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, calculateElement(gray))

    # 微处理去掉小的噪点
    dilation = cv2.dilate(binary, element1, iterations=1)
    binary = cv2.erode(dilation, element1, iterations=1)

    # 文字膨胀与腐蚀使其连成一个整体
    erosion = cv2.erode(binary, element2, iterations=1)
    dilation = cv2.dilate(erosion, element1, iterations=1)
    # show(erosion,'erosion')
    # show(dilation,'dilation')
    # 7. 存储中间图片
    # cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
    # cv2.imshow("binary", binary)
    # cv2.waitKey(0)
    #
    # cv2.namedWindow("dilation2", cv2.WINDOW_NORMAL)
    # cv2.imshow("dilation2", erosion)
    # cv2.waitKey(0)
    #
    # cv2.namedWindow("dilation2", cv2.WINDOW_NORMAL)
    # cv2.imshow("dilation2", dilation)
    # cv2.waitKey(0)

    cv2.destroyAllWindows()
    # sys.exit(0)

    return dilation

a=cv2.findContours(dilation = preprocess(gray, algos[0]), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(1)