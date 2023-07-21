import cv2  #######边缘检测.
import argparse
import imutils
import cv2

import numpy as np
kernel = np.ones((1, 5), np.uint8)
# img = cv2.imread('tmp99.png')
# img = cv2.imread('data/tq.jpg')

d='data/OIP-C.jpg'
d='data/tt3.jpg'
# d='99999.png'
# d='data/tt3.jpg'

# d='data/tq.jpg'
# d='data/tq2.png'


def  imgRotation(pathtoimg):
        print('旋转修复的图片是',pathtoimg)
        #图片自动旋正
        from PIL import Image
        img = Image.open(pathtoimg)
        path=pathtoimg
        new_img=cv2.imread(path)
        if hasattr(img, '_getexif') and img._getexif() != None:
            # 获取exif信息
            dict_exif = img._getexif()
            if 274 in dict_exif:
                if dict_exif[274] == 3:
                    #顺时针180
                    new_img = cv2.imread(path)
                    new_img=cv2.rotate(new_img,cv2.ROTATE_180)
            
                elif dict_exif[274] == 6:
                    #顺时针90°
                    new_img = cv2.imread(path)
                    new_img=cv2.rotate(new_img,cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif dict_exif[274] == 8:
                    #逆时针90°

                    new_img = cv2.imread(path)
                    new_img=cv2.rotate(new_img,cv2.ROTATE_90_CLOCKWISE)
        return new_img









img=imgRotation(d)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE) 
cv2.imwrite("img2.png", binary)   
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=15) # 二值化.
contours = cv2.findContours(binary,cv2.RETR_CCOMP  ,cv2.CHAIN_APPROX_SIMPLE)  # 参数说明;  https://docs.opencv.org/4.0.0/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71 
contours = imutils.grab_contours(contours) #适配cv2各个版本.
contours = sorted(contours, key = cv2.contourArea, reverse = True)
for i in contours:
    print(cv2.contourArea(i))




#=====step1:我们做id 卡片识别. 至少至少,需要1000像素包含整个id信息.也就是30**2
contours=[i for i in contours if cv2.contourArea(i)>1000]


#==========setp1.5 直接找到4个边界点.# 四角检测法!!!!!!!!!!!!!!!!!
if 1:
    print()
    tmp=contours[0]
    a=cv2.boundingRect(tmp)
    #======处理边缘锯齿. 首先去掉无效锯齿部分. 
    print()

    changdu=a[2]
    gaodu=a[3]
    print()



    result = cv2.pointPolygonTest(tmp, (0,0), False)
    result = cv2.pointPolygonTest(tmp, (1043,594), False)

    #======多边形进行填充. 然后算垂直投影.
    print()

    import numpy as np 
    tmp999=np.zeros_like(img)
    tmp2=cv2.fillPoly(tmp999, [tmp], (1, 1, 1))

    print()


    tmp2hang=np.sum(tmp2,axis=1)[:,0]

    #========改成算距离中心点最远的点.
    a_centerx=a[0]+a[2]/2
    a_centery=a[1]+a[3]/2
    print(a_centerx,a_centery)

    # tmp中距离中心点最远的点. 4个方向.
    tmp_juli=(tmp[:,:,0]-a_centerx)**2+(tmp[:,:,1]-a_centery)**2
    left,up,right,down=0,0,0,0
    leftdex,updex,rightdex,downdex=0,0,0,0
    tmp_juli=tmp_juli.flatten().tolist()
    for i in range(len(tmp_juli)):
        if tmp[i,0,0]<=a_centerx and    tmp[i,0,1]<=a_centery and tmp_juli[i]>left:
            left=tmp_juli[i]
            leftdex=i
            continue
        if tmp[i,0,0]<=a_centerx and    tmp[i,0,1]>=a_centery and tmp_juli[i]>up:
            up=tmp_juli[i]
            updex=i
            continue
        if tmp[i,0,0]>=a_centerx and   tmp[i,0,1]>=a_centery and tmp_juli[i]>down:
            down=tmp_juli[i]
            downdex=i
            continue
        if tmp[i,0,0]>=a_centerx and    tmp[i,0,1]<=a_centery and tmp_juli[i]>right:
            right=tmp_juli[i]
            rightdex=i
            continue
    print(1)

    k=[leftdex,updex,rightdex,downdex]
    left,up,right,down
    # debug
    for i in k:
        img=cv2.circle(img,tmp[i,0],40,(0,255,255),2)
    cv2.imwrite('debug1!!!!!!.png',img)
#####setp 1.6 仿射变化.


#====setp2: 利用ocr识别. 判断字高跟边界的距离来判断当前id卡是id卡本身边界还是id卡的真包含.



# from paddleocr import PaddleOCR, draw_ocr
# # use_angle_cls参数用于确定是否使用角度分类模型，即是否识别垂直方向的文字。
# ocr = PaddleOCR(use_angle_cls=True, use_gpu=False,

# lang='en',

# # det_model_dir="PaddleOCR/inference/ch_ppocr_server_v2.0_det_infer"  ,
# # rec_model_dir="PaddleOCR/inference/ch_ppocr_server_v2.0_rec_infer"  ,
# # cls_model_dir="PaddleOCR/inference/ch_ppocr_mobile_v2.0_cls_infer"  ,

# use_space_char=True

# )
# img_path = r'tmp.png'
# img_path = r'tmp99.png'
# result = ocr.ocr(img_path, cls=True)
# for line in result:
#     print(line)



















#=====================
if len(contours)>1:
    a=cv2.boundingRect(contours[0])
    print(a)
    print('比例:',a[2]/a[3])
if len(contours)>2:
    a=cv2.boundingRect(contours[1])
    print(a)
    print('比例:',a[2]/a[3])
binary=cv2.drawContours(img,contours,-1,(0,255,255),3)  
binary=cv2.rectangle(binary, (a[0],a[1]), (a[0]+a[2],a[1]+a[3]), (0, 255, 255), 3)
cv2.imwrite("img.png", binary)  
