#####=开始做黑白彩色识别.色素差. 后续看看有没有更靠谱的方法.
import cv2
import numpy as np


import cv2 as cv
path='data/tt2.jpg'
path='data_heibai/tt2.jpg'
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

#=======图片都统一变成三色图
#=======计算图像跟灰色图的色差
if len(origin_rotaed_img.shape)!=3:
    origin_rotaed_img=np.expand_dims(origin_rotaed_img,axis=2)
if len(image.shape)!=3:
    image=np.expand_dims(image,axis=2)


dif=np.abs(origin_rotaed_img-image).flatten()
dif=np.sum(dif)/len(dif)
print(dif)
if dif>50:
    print('当前是原件')
else:
    print('当前是复印件')












