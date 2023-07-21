# 三个需求
# 张博:
# 一个ocr

# 张博:
# 一个原件检测

# 张博:
# 一个抽取照片
# ocr还需要解决图片矫正问题，否则会导致识别率低





# 先做仿射变换矫正.# 仿射等于平移加旋转.

#======透视变换.  # 参考!!!https://blog.csdn.net/qq_22764813/article/details/120883528


#=========首先我们要学习cv2里面的坐标跟现实中的含义.
# 公式:   (x,y) 表示这个点到y轴距离x, 到图像上边界距离是y!!!!!!!!!!!
# 画图软件里面移动鼠标给的坐标值含义.跟上面cv2定义的是完全一样的.所以直接用即可.



import os
import cv2,math
import numpy as np
img_path = 'demo.png'
img = cv2.imread(img_path)
points_str = '145.10,263.40,1698.83,140.18,1424.10,909.50,484.30,1015.60'
points_str = '69,124,806,65,675,413,230,408' # 矩阵的四个角坐标.
 
# id_num_point = self.validate_label(id_num_point)  # 以左上顶点开始的顺时针box坐标



if 1:
    # 1.43 投影变换 (Projective mapping)
 
    h, w = img.shape[:2]  # 图片的高度和宽度
# '69,124,806,65,675,413,230,408'
    pointSrc = np.float32([[69,124], [806,65], [231,481], [675,432]])  # 原始图像中 4点坐标    .   (左上, 右上, 左下, 右下)


#=======changdu 和shuzhi 按照短边计算.
    chagndu=((pointSrc[1][0]-pointSrc[0][0])**2+(pointSrc[1][1]-pointSrc[1][1])**2)**0.5
    shuzhi=((pointSrc[2][0]-pointSrc[0][0])**2+(pointSrc[2][1]-pointSrc[0][1])**2)**0.5


    chagndu2=((pointSrc[3][0]-pointSrc[2][0])**2+(pointSrc[3][1]-pointSrc[2][1])**2)**0.5    
    shuzhi2=((pointSrc[3][0]-pointSrc[2][0])**2+(pointSrc[3][1]-pointSrc[2][1])**2)**0.5

    chagndu3=(chagndu+chagndu2)/2
    shuzhi3=(shuzhi+shuzhi2)/2


#========原始比例的话, 会丢失边界信息.所以要放缩矩形.
    #=======算每个边对于他整个延长线的比例.

    # 修改成高度. 这样来保持比例.

    chagndu=(max([(pointSrc[1][0]-pointSrc[0][0])**2,(pointSrc[1][1]-pointSrc[1][1])**2]))**0.5
    shuzhi=(max([(pointSrc[2][0]-pointSrc[0][0])**2,(pointSrc[2][1]-pointSrc[0][1])**2]))**0.5


    chagndu2=(max([(pointSrc[3][0]-pointSrc[2][0])**2,(pointSrc[3][1]-pointSrc[2][1])**2]))**0.5    
    shuzhi2=(max([(pointSrc[3][0]-pointSrc[2][0])**2,(pointSrc[3][1]-pointSrc[2][1])**2]))**0.5

    chagndu3=(chagndu+chagndu2)/2
    shuzhi3=(shuzhi+shuzhi2)/2








    chagndu=chagndu3
    shuzhi=shuzhi3


    #===
    a=(pointSrc[0][0]+pointSrc[2][0])/2
    b=(pointSrc[0][1]+pointSrc[1][1])/2
    pointDst= np.float32([
        
        [a,b], 
        [a+chagndu,b], 
        [a,b+shuzhi], 
     [a+chagndu,b+shuzhi] , 
        
        
        
        
        ])
    # pointDst = np.float32([[180,50], [w-180,50], [0,h-100], [w-1, h-100]])  # 变换图像中 4点坐标    (左上, 右上, 左下, 右下)
    # tmp=pointSrc[2].copy()
    # pointSrc[2]=pointSrc[3]
    # pointSrc[3]=tmp


    # tmp=pointDst[2].copy()
    # pointDst[2]=pointDst[3]
    # pointDst[3]=tmp
    MP = cv2.getPerspectiveTransform(pointSrc, pointDst)  # 计算投影变换矩阵 M
    imgP = cv2.warpPerspective(img, MP, (w, h))  # 用变换矩阵 M 进行投影变换

    save_img_path = 'demo2.png'
    cv2.imwrite(save_img_path, imgP)





