# 得到图片的边缘. # =======发现旋转预处理是必须的!!!!!!!!
# 测试发现二乙酯之后效果更差了!
import cv2
import numpy as np


import cv2 as cv
path='data/tt3.jpg'
path='data/OIP-C.jpg'
path='data/tq2.png'
# path='data/tq.jpg'
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
dic=['SERIALNUMBER','SERIALNUMBER:','IDNUMBER','IDNUMBER:','ID NUMBER','FULLNAMES','FULL NAMES','DATE OF BIRTH','DATEOFBIRTH','DATE OFBIRTH','DATEOF BIRTH','SEX','DISTRICTORBIRTH','DISTRICTOFBIRTH','DISTRICT OFBIRTH','DISTRICTORBIRTH', 'PLACEOFISSUE','PLACEOFISSUE','PLACE OF ISSUE','PLACEOF ISSUE','PLACE OFISSUE','DATE OFISSUE','DATEOFISSUE','DATE OF ISSUE',]
sav_dic=dic
for i in result:
 for j in dic:
    if i[1][0].replace(' ','')==j.replace(' ','') :
        remember[j]=i[0][0]
        break
    # elif 'SERIAL' in i[1][0]:
    #     remember['SERIALNUMBER']=i[0][0]
    #     break
for i in result:
    if 'KENYA' not in i[1][0] and i[1][0] not in dic and "HOLDER'S SIGN" not in i[1][0]:
        other[i[1][0]]=i[0][0]



        


print(1)

out={}




for  j in remember:
    if 'NUMBER' in j:
        minidis=float('inf')
        jiyi=0
        for i in other:
                
                
                
                    tmpdis=(other[i][0]-remember[j][0])**2+(other[i][1]-remember[j][1])**2
                    # gaodu=
                    if tmpdis<minidis  and  remember[j][0]<other[i][0] and remember[j][1]>other[i][1]:
                        minidis=tmpdis
                        jiyi=i
        if jiyi:
            out[j]=jiyi
#========要以坑为准, 把萝卜遍历跟坑比较才是正确逻辑.
for  j in remember:
    if 'NUMBER' not in j:
        minidis=float('inf')
        jiyi=0
        for i in other:
                
                
                
                    tmpdis=(other[i][0]-remember[j][0])**2+(other[i][1]-remember[j][1])**2
                    if tmpdis<minidis  and  remember[j][1]<other[i][1]:
                        minidis=tmpdis
                        jiyi=i
        if jiyi:
            out[j]=jiyi

# distance={} # 过来记忆最小距离,因为存在多匹配一个的情况, 这时候最优化. 让损失最小.
# for i in other:
#     minidis=float('inf')
#     jiyi=0
#     for j in remember:
#      if 'NUMBER'  not in i:
#         tmpdis=(other[i][0]-remember[j][0])**2+(other[i][1]-remember[j][1])**2
#         if tmpdis<minidis and  remember[j][1]<other[i][1]:
#             minidis=tmpdis
#             jiyi=j
            
#     if jiyi:#======在最后赋值时候存在一对多时候,所以要最优匹配.
#      if jiyi not in out:
#         out[jiyi]=i
#         distance[jiyi]=minidis
#      else:
#         if tmpdis<distance[jiyi]:
#             out[jiyi]=i
#             distance[jiyi]=minidis

print(2)





#===========添加容错:如果匹配到了一行. number
for i in result:
    if 'SERIAL' in i[1][0]:
        tmp=i[1][0]
        tmp2=tmp.replace('SERIAL','').replace('NUMBER','').replace(' ','')
        b=tmp.replace(tmp2,'')
        if len(tmp2)>0:
            try:
                aaa=int(tmp2)
                out[b]=tmp2
            except:
                pass
    if 'ID NUMBER' in i[1][0] or 'IDNUMBER' in i[1][0]:
        tmp=i[1][0]
        tmp2=tmp.replace('ID','').replace('NUMBER','').replace(' ','')
        b=tmp.replace(tmp2,'')
        if len(tmp2)>0:
            try:
                aaa=int(tmp2)
                out[b]=tmp2
            except:
                pass

print(out,'最终的匹配字典')
print('字典长度',len(out))


out2={}
for i in out:
    out2[i.replace(' ','').replace(':','')]=out[i]
out=out2
#====================再次补漏
if 1:   


        #======预处理.
        result.sort(key=lambda x: x[0][0][1])
        result2=[]
        for i in result:
            result2.append([i[1][0],i[0][0],i[0][2],i[1][1]])
        result2=result2[2:]
        # result2=[i for i in result2 if 'NUMBER' not in i] # 过滤.?????????不应该弄好像.
        print('++++++++++++++++++++++++++')
        print(result2)
        #======找到大数字的高度.
        shuzi=['1','2','3','4','5','6','7','8','9','0']
        withnumber=[]
        for dex,i in enumerate(result2):
            
                if set(i[0]) &   set(shuzi):
                    withnumber.append(i)
        withnumber=withnumber[:2]# 钱2个必识别出来.
        withnumber.sort(key=lambda x: x[1][0])
        shuzi=['1','2','3','4','5','6','7','8','9','0']
        # 如果两个数字没识别出来.
        if 'SERIALNUMBER' not in out:
            #========把钱4个结果拿出来,直接复制数字即可!这里还用字典,防止多次重复.+
            out['SERIALNUMBER']=''.join([i for i in withnumber[0][0] if i in shuzi])
            print(1)
            pass
        if 'IDNUMBER' not in out:
                    #========把钱4个结果拿出来,直接复制数字即可!这里还用字典,防止多次重复.+
                    out['IDNUMBER']=''.join([i for i in withnumber[1][0] if i in shuzi])
                    print(1)














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
 if 0:
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
        print('没检测到人脸')
    # vis_face(img_bg,bboxs,landmarks, save_name)


# 2023-07-14,0点06  考虑接一个英文分词工具,======基本不行, 分不开黑人名字.
#  或者优化一个ocr里面空格识别.



#-===========引入微软模型, 对名字进行修复.




out_for_fullname=0
for i in result:
    if 'FULLNAMES' in out and i[1][0]==out['FULLNAMES']:
        out_for_fullname=i[0]
print('名字对应的位置是左上,右上,右下,左下',out_for_fullname)


try:
    out_for_fullname2=0
    for i in result:
        if i[1][0]=='FULLNAMES' or i[1][0]=='FULL NAMES':
            out_for_fullname2=i[0]
    print('名字fullname标志对应的位置是左上,右上,右下,左下',out_for_fullname2)
    dangezifuchangdu =( out_for_fullname2[1][0]-out_for_fullname2[0][0])/9
    print(dangezifuchangdu,'单字符长度')
    left=out_for_fullname2[0][0]-dangezifuchangdu
    print('图片位置left',left)
    right=left+200
    out_for_fullname3=0
    for i in result:
        if i[1][0]=='SEX':
            out_for_fullname3=i[0]
    print('名字sex标志对应的位置是左上,右上,右下,左下',out_for_fullname3)
    if out_for_fullname3:
        print(dangezifuchangdu,'单字符长度')
        right=out_for_fullname3[0][0]-dangezifuchangdu*2
        print('图片位置right',right)

except:
    left=0
    right=left+200







try:
    dangezifugaodu =( out_for_fullname2[3][1]-out_for_fullname2[0][1])
    print('字符高度',dangezifugaodu)
    up=out_for_fullname2[0][1]+dangezifugaodu*4
    print('图片位置up',up)
    down=out_for_fullname2[0][1]+dangezifugaodu*21
    print('图片位置down',down)
except:
        up=0





try:
        out_for_fullname4=0
        for i in result:
            if 'HOLDER' in i[1][0]:
                out_for_fullname4=i[0]
        print('holder weizhi ',out_for_fullname4)

        down=out_for_fullname4[0][1]+dangezifugaodu*3.7
except:
    down=50








#==========下面我们用高精度模型修复名字问题.!!!!!!!!!!!!!!!!!!!!
if 1:
    
        # ========只能识别单行的. 先不用.

        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from PIL import Image
        import requests

        # load image from the IAM database (actually this model is meant to be used on printed text)
        url = 'tmp99.png'
        image = Image.open(url).convert("RGB")

        tupiangaodu=out_for_fullname[2][1]-out_for_fullname[0][1]

        bili=0.2
        d=[out_for_fullname[0][0],out_for_fullname[0][1]-tupiangaodu*bili,out_for_fullname[2][0],out_for_fullname[2][1]+tupiangaodu*bili]
        print('切割点',d)
        image=image.crop(d)






        image.save('tmp100.png')





        # d='/mnt/e/trocr-base_printed'
        d='microsoft/trocr-base-printed' #测试后还是这个版本最稳.
        # d='microsoft/trocr-small-printed'
        processor = TrOCRProcessor.from_pretrained(d)
        # model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
        model = VisionEncoderDecoderModel.from_pretrained(d)
        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_text)



        out['FULLNAMES']=generated_text
        print('修复后的out',out)
        print('修复后的out长度',len(out))

        #========转化为数组模式:
        out2={}
        for i in out:
            out2[i.replace(' ','').replace(':','')]=out[i]
        out3=out2.values
        out3=[]
        for i in ['SERIALNUMBER','SERIALNUMBER:','IDNUMBER','IDNUMBER:','ID NUMBER','ID NUMBER:','FULLNAMES','FULL NAMES','DATE OF BIRTH','DATEOFBIRTH','DATE OFBIRTH','DATEOF BIRTH','SEX','DISTRICTORBIRTH','DISTRICTOFBIRTH','DISTRICT OFBIRTH','DISTRICTORBIRTH', 'PLACEOFISSUE','PLACEOFISSUE','PLACE OF ISSUE','PLACEOF ISSUE','PLACE OFISSUE','DATE OFISSUE','DATEOFISSUE','DATE OF ISSUE',]:# 注意顺序.
            if i in out2:
                out3.append(out2[i])
        print('最终数组输出模式',out3)
        out=out3

#-===============头像像素位置切个./


url = 'tmp99.png'
image = Image.open(url).convert("RGB")



url = 'tmp99.png'
image = Image.open(url).convert("RGB")

touxiang=[left,up,right,down]
image=image.crop(touxiang)
image.save('tmp101.png')





#===============下面处理非常糊的图片的情况.
if len(out)<8:
    pass # ========进行模糊处理.

    print('下面进行模糊匹配!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    #导入库
    import cv2
    import skimage.filters as af
    import  skimage.filters
    import matplotlib.pyplot as plt
    from PIL import  Image
    from PIL import  ImageFilter
    from PIL.ImageFilter import  FIND_EDGES,EDGE_ENHANCE,EDGE_ENHANCE_MORE,SHARPEN


    if 0:
        im=Image.open("tmp99.png")
        im_01=im.filter(FIND_EDGES)
        im_02=im.filter(EDGE_ENHANCE)
        im_03=im.filter(EDGE_ENHANCE_MORE)
        im_04=im.filter(SHARPEN)








        im_01.save('tmp9991.png')
        im_02.save('tmp9992.png')
        im_03.save('tmp9993.png')
        im_04.save('tmp9994.png')



        im=cv2.imread('tmp99.png')
        img=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        img_gaussianBlur = cv2.GaussianBlur(img, (3, 3), 1)
        # 锐化图像=原始图像+(原始图像-模糊图像)
        im_fun_01=img+(img-img_gaussianBlur)*10
        im_fun_02 = img + (img - img_gaussianBlur) * 20
        im_fun_03 = img + (img - img_gaussianBlur) * 30

        cv2.imwrite('tmp99971.png',im_fun_01)
        cv2.imwrite('tmp99972.png',im_fun_02)
        cv2.imwrite('tmp99973.png',im_fun_03)





        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        im = cv2.filter2D(im, -1, kernel)
        cv2.imwrite('tmp99974.png',im)





# ======超分:https://blog.csdn.net/jiazhen/article/details/115274863
    if 0:
        import cv2
        im=cv2.imread('tmp99.png')
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        path = "ESPCN_x4.pb" 
        sr.readModel(path) 
        sr.setModel("espcn", 4) # set the model by passing the value and the upsampling ratio
        result = sr.upsample(im) # upscale the input image
        cv2.imwrite('tmp99975.png',result)


    print(result,'即系的东西')


    result.sort(key=lambda x: x[0][0][1])
    result2=[]
    for i in result:
        result2.append([i[1][0],i[0][0],i[0][2],i[1][1]])
    result2=result2[2:]
    result2=[i for i in result2 if 'NUMBER' not in i] # 过滤.?????????不应该弄好像.
    print('++++++++++++++++++++++++++')
    print(result2)
    #======找到大数字的高度.
    shuzi=['1','2','3','4','5','6','7','8','9','0']

    for dex,i in enumerate(result2):
        
            if set(i[0]) &   set(shuzi):
                break
    dashuzigaodu=i[2][1]-i[1][1]
    print(dashuzigaodu)
    diyigegaodu=i[1][1]

    ##setp1抽取数字
    fanwei=[diyigegaodu-0.3*dashuzigaodu,diyigegaodu+0.3*dashuzigaodu]
    cnt=[]
    for dex,i in enumerate(result2):
            tmpgaodu =i[1][1]
            if set(i[0]) &   set(shuzi) and fanwei[0]<tmpgaodu<fanwei[1]:
                cnt.append(i[0])
    #==========如果serial不在第一个位置,就换.
    cnt2=[]
    for dex,i in enumerate(cnt):

        if 'SERIAL' in i:
            cnt2.append(i)
            for j in cnt:
                if j not in cnt2:
                    cnt2.append(j)
            break
    for i in range(len(cnt2)):
        cnt2[i]=cnt2[i].replace('SERIAL','').replace('NUMBER','').replace(' ','').replace(':','').replace('IN','')
    cnt=cnt2
    print(1)
    if 0:#抽取失败走另外的方案.
        cnt=[]

        for i in result:
            if 'SERIAL' in i[1][0]:
                tmp=i[1][0]
                tmp2=tmp.replace('SERIAL','').replace('NUMBER','').replace(' ','').replace(':','')
                b=tmp.replace(tmp2,'')
                if len(tmp2)>0:
                    try:
                        aaa=int(tmp2)
                        cnt.append(tmp2)
                    except:
                        pass
            if 'ID NUMBER' in i[1][0] or 'IDNUMBER' in i[1][0]:
                tmp=i[1][0]
                tmp2=tmp.replace('ID','').replace('NUMBER','').replace(' ','')
                b=tmp.replace(tmp2,'')
                if len(tmp2)>0:
                    try:
                        aaa=int(tmp2)
                        cnt.append(tmp2)
                    except:
                        pass
        fanwei=[diyigegaodu-0.1*dashuzigaodu,diyigegaodu+0.1*dashuzigaodu]
        
        for dex,i in enumerate(result2):
                tmpgaodu =i[1][1]
                if set(i[0]) &   set(shuzi) and fanwei[0]<tmpgaodu<fanwei[1]:
                    cnt.append(i[0])

        print(1)











    #抽取名字.

    fanwei=[diyigegaodu+2.1*dashuzigaodu,diyigegaodu+3.1*dashuzigaodu]

    for dex,i in enumerate(result2):
            tmpgaodu =i[1][1]
            if  fanwei[0]<tmpgaodu<fanwei[1]:
                cnt.append(i[0])
    print(1)
    #==========名字强化算法.
    if 0:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from PIL import Image
        import requests

        # load image from the IAM database (actually this model is meant to be used on printed text)
        url = 'tmp99.png'
        image = Image.open(url).convert("RGB")

        tupiangaodu=out_for_fullname[2][1]-out_for_fullname[0][1]

        bili=0.1
        d=[out_for_fullname[0][0],out_for_fullname[0][1]-tupiangaodu*bili,out_for_fullname[2][0],out_for_fullname[2][1]+tupiangaodu*bili]
        print('切割点',d)
        image=image.crop(d)






        image.save('tmp100.png')





        # d='/mnt/e/trocr-base_printed'
        d='microsoft/trocr-small-printed'
        processor = TrOCRProcessor.from_pretrained(d)
        # model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
        model = VisionEncoderDecoderModel.from_pretrained(d)
        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_text)





















    #抽取生日.
    fanwei=[diyigegaodu+3.6*dashuzigaodu,diyigegaodu+11.5*dashuzigaodu]
    print(fanwei)

    for dex,i in enumerate(result2):
            tmpgaodu =i[1][1]
            if  fanwei[0]<tmpgaodu<fanwei[1] and i[3]>0.9 and i[0] not in sav_dic:
                cnt.append(i[0])






#===========添加容错:如果匹配到了一行. number







    print(1)
    print(cnt,'最终结果.')
    print(len(cnt),'最终结果数量.')
    out=cnt







if len(out)==8:
    #==========最终根据字符数量再修复:
    if len(out[4])==4:
        out[4]='MALE'
    else:
        out[4]='FEMALE'
print('第二次修复之后的结果',out)