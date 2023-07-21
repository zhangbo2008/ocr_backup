# 得到图片的边缘. # =======发现旋转预处理是必须的!!!!!!!!
# 测试发现二乙酯之后效果更差了!
import cv2
import numpy as np


import cv2 as cv
path='data/tt.jpg'
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
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False,lang='en',



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

#========解析:
result=result[0]
for i in result:



    #======='进行一些replace'
    if i=='DISTRICTORBIRTH':
        i='DISTRICT OF BIRTH'
    print(i[1][0])



#=========利用首字母距离法进行排序.也就是左上.



remember={}
other={}
dic=['SERIALNUMBER','IDNUMBER','FULLNAMES','DATE OF BIRTH','SEX','DISTRICTORBIRTH','DISTRICTOFBIRTH', 'PLACEOFISSUE','DATE OFISSUE',"HOLDER'S SIGN",'PLACE OF ISSUE','PLACEOF ISSUE','PLACE OFISSUE','DATEOFISSUE','DATE OF ISSUE','PLACEOFISSUE','DISTRICTORBIRTH']
for i in result:
 for j in dic:
    if i[1][0]==j :
        remember[j]=i[0][0]
        break
    elif 'SERIAL' in i[1][0]:
        remember['SERIALNUMBER']=i[0][0]
        break
for i in result:
    if 'KENYA' not in i[1][0] and i[1][0] not in dic:
        other[i[1][0]]=i[0][0]

print(1)

out={}
for i in other:
    minidis=float('inf')
    jiyi=0
    for j in remember:
        tmpdis=(other[i][0]-remember[j][0])**2+(other[i][1]-remember[j][1])**2
        if tmpdis<minidis:
            minidis=tmpdis
            jiyi=j
    if jiyi:
        out[jiyi]=i

print(2)

print(out,'最终的匹配字典')










#=========人俩检测:

# cv2精度不行弃用!!!!!!
# import cv2 as cv

# img = origin_rotaed_img
# """
# 讲解：
#     人脸检测没有与皮肤、颜色在图像里。这个 haarcascade本质上是寻找一个物体在图片中，使用边缘edge去决定是否有人脸
# """
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # 实际所做：解析xml文件，读取，再保存到这个变量里。
# haarcascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# # 参数：  3：the number of the neighbor of the rectangle should be called  a face
# face_rect = haarcascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
# print("图片中发现的人脸的数量: ", len(face_rect))
# # 循环face_rect，将每一个人脸都画上一个矩形
# for (x,y,w,h) in face_rect:
#     cv.rectangle(img, (x,y), (x+w, y+h), color=(0,255,0))
# # cv.imshow("人脸检测", img)

# cv2.imwrite('faceimg.png',img)








import cv2
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import vis_face




if __name__ == '__main__':

    pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt", r_model_path="./original_model/rnet_epoch.pt", o_model_path="./original_model/onet_epoch.pt", use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    img = origin_rotaed_img
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #b, g, r = cv2.split(img)
    #img2 = cv2.merge([r, g, b])

    bboxs, landmarks = mtcnn_detector.detect_face(img)
    # print box_align
    save_name = 'r_4.jpg'
    if len(bboxs)>0:
        for i in bboxs:
            i=[int(j) for j in i]
            print(i,'检测到的人脸矿是.')
            tmp_save=origin_rotaed_img[i[1]:i[3],i[0]:i[2],]
            cv2.imwrite('tmp_save.png', tmp_save)
            break
    else:
        print('没检测到')
    # vis_face(img_bg,bboxs,landmarks, save_name)
