#####=开始做背面的ocr
import cv2
import numpy as np


import cv2 as cv
path='data/tt2.jpg'
# path='data/rot.png'

#==========先用图像自带的修正:

#==========图像自动修正.
def  imgRotation(pathtoimg):
    #图片自动旋正
    from PIL import Image
    img = Image.open(pathtoimg)
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









image=imgRotation(path)

origin_rotaed_img=image
image=cv.cvtColor(image,cv.COLOR_RGB2GRAY)

cv2.imwrite('tmp99.png',origin_rotaed_img)
if 1: # 两种去噪方式. 腐蚀和膨胀!!!!!!!!
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel,iterations=3) #形态学关操作
    image = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel,iterations=3)  #形态学开操作
    # img_closed = cv2.erode(mg_closed, None, iterations=9)    #腐蚀
    # img_closed = cv2.dilate(img_closed, None, iterations=9)  # 膨胀










threshold = cv2.threshold(image, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)[1]











canny = cv2.Canny(threshold, 100, 150)
# show(canny, "canny")
kernel = np.ones((3, 3), np.uint8)
dilate = cv2.dilate(canny, kernel, iterations=3)

# show(dilate, "dilate")




# cv.imshow('THRESH_BINRY',binary)
cv2.imwrite('binary8.jpg',dilate)



from PIL import Image
img = Image.open(path)
print(1)






#============处理后续ocr
# 转换为灰度图
gray = cv2.cvtColor(origin_rotaed_img, cv2.COLOR_BGR2GRAY)
# 二值化处理
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imwrite('tmp.png',thresh)

# import pytesseract  # 安装:https://blog.csdn.net/qq_44314841/article/details/105602017   # https://tesseract-ocr.github.io/tessdoc/Installation.html



# cv2.imwrite('ttttttt.png',thresh)
# # 使用pytesseract识别
# text = pytesseract.image_to_string(thresh)
# print(text)

# import easyocr
# reader = easyocr.Reader([ 'en'])
# result = reader.readtext(thresh)

# print(result)


#ceshi paddleocr # python 3.9.1  # https://zhuanlan.zhihu.com/p/380142530
# pip3 install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
# -i https://mirror.baidu.com/pypi/simple



from paddleocr import PaddleOCR, draw_ocr
# use_angle_cls参数用于确定是否使用角度分类模型，即是否识别垂直方向的文字。
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False,

lang='en',

# det_model_dir="PaddleOCR/inference/ch_ppocr_server_v2.0_det_infer"  ,
# rec_model_dir="PaddleOCR/inference/ch_ppocr_server_v2.0_rec_infer"  ,
# cls_model_dir="PaddleOCR/inference/ch_ppocr_mobile_v2.0_cls_infer"  ,

use_space_char=True

)
img_path = r'tmp.png'
img_path = r'tmp99.png'
result = ocr.ocr(img_path, cls=True)
for line in result:
    print(line)
result=result[0]
result.sort(key=lambda x: x[0][0][1])
print(result)

out={}
out[result[0][1][0]]=result[1][1][0]
out[result[2][1][0]]=result[3][1][0]
out[result[4][1][0]]=result[5][1][0]
out[result[6][1][0]]=result[7][1][0]

result=result[8:]
qiege=0
for dex,i in enumerate(result):
    if '<<' in i and "ID" in i:
        qiege=dex
result=result[qiege:]
tmp=''.join([i[1][0] for i in result])
print(tmp)
out['code']=tmp

print('最后的out',out)


