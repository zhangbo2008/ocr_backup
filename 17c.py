#3  使用paddleocr新的配置参数, 达到最好效果.# 优化一下速度.


# 得到图片的边缘. # =======发现旋转预处理是必须的!!!!!!!!
# 测试发现二乙酯之后效果更差了! 整个代码韩淑华.



zhengmian='data/20230608/27-1.jpg'
beimian=  'data/20230608/27-2.jpg'
import cv2
import numpy as np

from PIL import Image


import multiprocessing
aa=multiprocessing.cpu_count()
from paddleocr import PaddleOCR, draw_ocr
# use_angle_cls参数用于确定是否使用角度分类模型，即是否识别垂直方向的文字。
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_dilation=True,use_mp=True,total_process_num=aa   ,show_log=False

)





import time
a9999999999999=time.time()








if 1:
    import cv2 as cv
    import cv2

    path=zhengmian
    #print('chulitupian ',path)
    # path='data/tq.jpg'
    # path='data/rot.png'

    #==========先用图像自带的修正:

    #==========图像自动修正.
    def  imgRotation(pathtoimg):
        #print('旋转修复的图片是',pathtoimg)
        #图片自动旋正
        from PIL import Image
        img = Image.open(pathtoimg)
        # new_img=cv2.imread(path)
        new_img=cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        if hasattr(img, '_getexif') and img._getexif() != None:
            # 获取exif信息
            dict_exif = img._getexif()
            if 274 in dict_exif:
                if dict_exif[274] == 3:
                    #顺时针180
           
                    new_img=cv2.rotate(new_img,cv2.ROTATE_180)
            
                elif dict_exif[274] == 6:
                    #顺时针90°

                    new_img=cv2.rotate(new_img,cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif dict_exif[274] == 8:
                    #逆时针90°

      
                    new_img=cv2.rotate(new_img,cv2.ROTATE_90_CLOCKWISE)
        return new_img









    image=imgRotation(path)

    origin_rotaed_img=image
    image=cv.cvtColor(image,cv.COLOR_RGB2GRAY)

    # cv2.imwrite('tmp99.png',origin_rotaed_img)
    if 0: # 两种去噪方式. 腐蚀和膨胀!!!!!!!!
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        img_closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel,iterations=3) #形态学关操作
        image = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel,iterations=3)  #形态学开操作
        # img_closed = cv2.erode(mg_closed, None, iterations=9)    #腐蚀
        # img_closed = cv2.dilate(img_closed, None, iterations=9)  # 膨胀










    threshold = cv2.threshold(image, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)[1]










    if 0:#边缘检测.
        canny = cv2.Canny(threshold, 100, 150)
        # show(canny, "canny")
        kernel = np.ones((3, 3), np.uint8)
        dilate = cv2.dilate(canny, kernel, iterations=3)

        # show(dilate, "dilate")




        # cv.imshow('THRESH_BINRY',binary)
        cv2.imwrite('binary8.jpg',dilate)



        from PIL import Image
        img = Image.open(path)
        #print(1)






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
    # #print(text)

    # import easyocr
    # reader = easyocr.Reader([ 'en'])
    # result = reader.readtext(thresh)

    # #print(result)


    #ceshi paddleocr # python 3.9.1  # https://zhuanlan.zhihu.com/p/380142530
    # pip3 install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
    # -i https://mirror.baidu.com/pypi/simple

    

    img_path = r'tmp99.png'
    result = ocr.ocr(origin_rotaed_img, cls=True)
    # for line in result:
        #print(line)

    #========解析:
    result=result[0]
    for i in result:



        #======='进行一些replace'
        if i=='DISTRICTORBIRTH':
            i='DISTRICT OF BIRTH'
        #print(i[1][0])



    #=========利用首字母距离法进行排序.也就是左上.


    remember={}
    other={}
    dic=['SERIALNUMBER','SERIALNUMBER:','IDNUMBER','IDNUMBER:','ID NUMBER','FULLNAMES','DATE OF BIRTH','DATEOFBIRTH','DATEORBIRTH','DATE OFBIRTH','DATEOF BIRTH','SEX','DISTRICTORBIRTH','DISTRICTOFBIRTH','DISTRICT OFBIRTH','DISTRICTORBIRTH', 'PLACEOFISSUE','PLACEOFISSUE','PLACE OF ISSUE','PLACEOF ISSUE','PLACE OFISSUE','DATE OFISSUE','DATEOFISSUE','DATE OF ISSUE',]
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



            


    #print(1)

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

    #print(2)





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
            # #print('++++++++++++++++++++++++++')
            # #print(result2)
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
                #print(1)
                pass
            if 'IDNUMBER' not in out:
                        #========把钱4个结果拿出来,直接复制数字即可!这里还用字典,防止多次重复.+
                        out['IDNUMBER']=''.join([i for i in withnumber[1][0] if i in shuzi])
                        #print(1)

    #print(out,'第一次匹配结果.')
    out2=[]
    out3=[]
    for j in sav_dic:
          for i in out:
            if i.replace(' ','')==j.replace(' ',''):
                
                if i not in out2:
                    out2.append(i)
                    out3.append(out[i])
    #print(1,out2)
    out=out3














    import cv2
    from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
    from mtcnn.core.vision import vis_face




    if 1:
     if 0:
        pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt", r_model_path="./original_model/rnet_epoch.pt", o_model_path="./original_model/onet_epoch.pt", use_cuda=False)
        mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

        img = origin_rotaed_img
        img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #b, g, r = cv2.split(img)
        #img2 = cv2.merge([r, g, b])

        bboxs, landmarks = mtcnn_detector.detect_face(img)
        # #print box_align
        save_name = 'r_4.jpg'
        if len(bboxs)>0:
            for i in bboxs:
                i=[int(j) for j in i]
                #print(i,'检测到的人脸矿是.')
                tmp_save=origin_rotaed_img[i[1]:i[3],i[0]:i[2],]
                cv2.imwrite('tmp_save.png', tmp_save)
                break
        else:
            pass
            #print('没检测到人脸')
        # vis_face(img_bg,bboxs,landmarks, save_name)


    # 2023-07-14,0点06  考虑接一个英文分词工具,======基本不行, 分不开黑人名字.
    #  或者优化一个ocr里面空格识别.





     
    #print('第一个方案定位的结果',fangfa1out)
    #===============下面处理非常糊的图片的情况.
    if 1:
        fangfa1out=out
        pass # ========进行模糊处理.

        #print('下面进行模糊匹配!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        out={}
        #导入库
        import cv2
        import skimage.filters as af
        import  skimage.filters
        import matplotlib.pyplot as plt
        from PIL import  Image
        from PIL import  ImageFilter
        from PIL.ImageFilter import  FIND_EDGES,EDGE_ENHANCE,EDGE_ENHANCE_MORE,SHARPEN


        if 0:

            im_01=im.filter(FIND_EDGES)
            im_02=im.filter(EDGE_ENHANCE)
            im_03=im.filter(EDGE_ENHANCE_MORE)
            im_04=im.filter(SHARPEN)












            img_gaussianBlur = cv2.GaussianBlur(img, (3, 3), 1)
            # 锐化图像=原始图像+(原始图像-模糊图像)
            im_fun_01=img+(img-img_gaussianBlur)*10
            im_fun_02 = img + (img - img_gaussianBlur) * 20
            im_fun_03 = img + (img - img_gaussianBlur) * 30






            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            im = cv2.filter2D(im, -1, kernel)






    # ======超分:https://blog.csdn.net/jiazhen/article/details/115274863
        

#===============第二个方案的开始.................
        result.sort(key=lambda x: x[0][0][1])
        result2=[]
        for i in result:
            result2.append([i[1][0],i[0][0],i[0][2],i[1][1]])
        result2=result2[2:]
        # result2=[i for i in result2 if 'NUMBER' not in i] # 过滤.?????????不应该弄好像.

        #======找到大数字的高度.
        shuzi=['1','2','3','4','5','6','7','8','9','0']

        for dex,i in enumerate(result2):
            
                if set(i[0]) &   set(shuzi):
                    break
        dashuzigaodu=i[2][1]-i[1][1]
        diyige=i
        #大数字高度进行修复.
        if 1:
            result.sort(key=lambda x: x[0][0][1])
            result2=[]
            for i in result:
                result2.append([i[1][0],i[0][0],i[0][2],i[1][1]])
            result2=result2[2:]
            # result2=[i for i in result2 if 'NUMBER' not in i] # 过滤.?????????不应该弄好像.
 
            #======找到大数字的高度.
            shuzi=['1','2','3','4','5','6','7','8','9','0']
            withnumber=[]
            for dex,i in enumerate(result2):
                
                    if set(i[0]) &   set(shuzi):
                        withnumber.append(i)
            withnumber=withnumber[:]# 钱2个必识别出来.
            withnumber.sort(key=lambda x: x[1][0])

            dashuzigaodu= sum([i2[2][1]-i2[1][1] for i2 in withnumber])/len(withnumber)
           







        # #print(dashuzigaodu)
        diyigegaodu=diyige[1][1]

        ##setp1抽取数字
        fanwei=[diyigegaodu-0.1*dashuzigaodu,diyigegaodu+0.1*dashuzigaodu]
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
                
                for i in range(len(cnt2)):
                    cnt2[i]=cnt2[i].replace('SERIAL','').replace('NUMBER','').replace(' ','').replace(':','').replace('IN','')
                cnt=cnt2
                break
        if 1:
            #======预处理.
            result.sort(key=lambda x: x[0][0][1])
            result2=[]
            for i in result:
                result2.append([i[1][0],i[0][0],i[0][2],i[1][1]])
            result2=result2[2:]
            # result2=[i for i in result2 if 'NUMBER' not in i] # 过滤.?????????不应该弄好像.
 
            #======找到大数字的高度.
            shuzi=['1','2','3','4','5','6','7','8','9','0']
            withnumber=[]
            for dex,i in enumerate(result2):
                
                    if set(i[0]) &   set(shuzi):
                        withnumber.append(i)
            withnumberall=withnumber
            withnumber=withnumber[:2]# 钱2个必识别出来.
            withnumber.sort(key=lambda x: x[1][0])
            shuzi=['1','2','3','4','5','6','7','8','9','0']
            # 如果两个数字没识别出来.
            if 'SERIALNUMBER' not in out:
                #========把钱4个结果拿出来,直接复制数字即可!这里还用字典,防止多次重复.+
                out['SERIALNUMBER']=''.join([i for i in withnumber[0][0] if i in shuzi])
       
                pass
            if 'IDNUMBER' not in out:
                        #========把钱4个结果拿出来,直接复制数字即可!这里还用字典,防止多次重复.+
                        out['IDNUMBER']=''.join([i for i in withnumber[1][0] if i in shuzi])
                
            cnt=[out['SERIALNUMBER'],out['IDNUMBER']]


     

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
                        break











        #========
        # name
        namexini=0
        fanwei=[diyigegaodu+2.3*dashuzigaodu,diyigegaodu+3.3*dashuzigaodu]
        #print(fanwei)
        for j2 in result2:
            if 'NAME' in j2[0]:
                fanwei=[j2[2][1]-0.12*dashuzigaodu,j2[2][1]+1.4*dashuzigaodu]
                break
        for dex,i in enumerate(result2):
                tmpgaodu =i[1][1]
                if  fanwei[0]<tmpgaodu<fanwei[1]  and i[0] not in sav_dic:# 排除那些标志位.
                    cnt.append(i[0])
                    break
        namexini=i














                    
        #print(1)
        #==========名字强化算法.
        if 0:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            from PIL import Image
            import requests

            # load image from the IAM database (actually this model is meant to be used on #printed text)

            image = Image.open(url).convert("RGB")

            tupiangaodu=out_for_fullname[2][1]-out_for_fullname[0][1]

            bili=0.1
            d=[out_for_fullname[0][0],out_for_fullname[0][1]-tupiangaodu*bili,out_for_fullname[2][0],out_for_fullname[2][1]+tupiangaodu*bili]
            #print('切割点',d)
            image=image.crop(d)






            # image.save('tmp100.png')





            # d='/mnt/e/trocr-base_#printed'
            d='microsoft/trocr-base-#printed'
            processor = TrOCRProcessor.from_pretrained(d)
            # model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-#printed')
            model = VisionEncoderDecoderModel.from_pretrained(d)
            pixel_values = processor(images=image, return_tensors="pt").pixel_values

            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            #print(generated_text)





















        #抽取生日.
        fanwei=[diyigegaodu+4.03*dashuzigaodu,diyigegaodu+5.23*dashuzigaodu]
        #print(fanwei)
        for j2 in result2:
            if 'DATE' in j2[0] or 'BIRTH'in j2[0]:
                fanwei=[j2[2][1]-0.1*dashuzigaodu,j2[2][1]+1.1*dashuzigaodu]
                break
        for dex,i in enumerate(result2):
                tmpgaodu =i[1][1]
                if  fanwei[0]<tmpgaodu<fanwei[1]  and i[0] not in sav_dic:# 排除那些标志位.
                    cnt.append(i[0])
                    break
        #抽取sex.
        fanwei=[diyigegaodu+5.65*dashuzigaodu,diyigegaodu+6.85*dashuzigaodu]
        #print(fanwei)
        for j2 in result2:
            if 'SEX' in j2[0]:
                fanwei=[j2[2][1]-0.1*dashuzigaodu,j2[2][1]+1.1*dashuzigaodu]
                break
        for dex,i in enumerate(result2):
                tmpgaodu =i[1][1]
                if  fanwei[0]<tmpgaodu<fanwei[1]  and i[0] not in sav_dic:# 排除那些标志位.
                    cnt.append(i[0])
                    break
        #抽取d.
        fanwei=[diyigegaodu+7.34*dashuzigaodu,diyigegaodu+8.64*dashuzigaodu]
        #print(fanwei)

        for j2 in result2:
            if 'SEX' in j2[0]:
                fanwei=[j2[2][1]+1.5*dashuzigaodu,j2[2][1]+2.7*dashuzigaodu]
                break
        for j2 in result2:
            if 'DISTRI' in j2[0]:
                fanwei=[j2[2][1]-0.1*dashuzigaodu,j2[2][1]+1.1*dashuzigaodu]
                break
        
        for dex,i in enumerate(result2):
                tmpgaodu =i[1][1]
                if  fanwei[0]<tmpgaodu<fanwei[1]  and i[0] not in sav_dic:# 排除那些标志位.
                    cnt.append(i[0])
                    break



        #抽取place.
        fanwei=[diyigegaodu+9.01*dashuzigaodu,diyigegaodu+10.81*dashuzigaodu]
        #print(fanwei)

        for j2 in result2:
            if 'SEX' in j2[0]:
                fanwei=[j2[2][1]+3.6*dashuzigaodu,j2[2][1]+4.6*dashuzigaodu]
                break
        for j2 in result2:
            if 'DISTRI' in j2[0]:
                fanwei=[j2[2][1]+1.46*dashuzigaodu,j2[2][1]+2.46*dashuzigaodu]
                break
        for j2 in result2:
            if 'PLACE' in j2[0]:
                fanwei=[j2[2][1]-0.1*dashuzigaodu,j2[2][1]+1.1*dashuzigaodu]
                break
        for dex,i in enumerate(result2):
                tmpgaodu =i[1][1]
                if  fanwei[0]<tmpgaodu<fanwei[1]  and i[0] not in sav_dic:# 排除那些标志位.
                    cnt.append(i[0])
                    break
        #print(222222222222)



#抽取date.
        cnt.append(withnumberall[-1][0])
        #print()










    #===========添加容错:如果匹配到了一行. number







        #print(1)
        #print(cnt,'最终结果.')
        #print(len(cnt),'最终结果数量.')
        out=cnt
    fangfa2out=out



    if (len(fangfa1out)>=len(fangfa2out)):#======我们优先相信高精度版本.
        out=fangfa1out
    if type(out)==type(dict()):
        out=list(out.values())

    if 'MALE' not in out and  'FEMALE' not in out:
        if len(out)>=8:
            #==========最终根据字符数量再修复:
            if len(out[4])==4 or out[4][:4]=='MALE':
                out[4]=   'MALE'
            else:
                out[4]= 'FEMALE'
    out2=[]
    for i in out:
        if i not in out2:
            out2.append(i)
    out=out2

    out_zhengmian=out
    #print('第二次修复之后的结果',out)
    #print('第二次修复之后的结果',len(out))

#==========================下面统一进行头像切割:



    # if  left!=0 and right!=200:
    #     touxiang=[left,up,right,down]
    #     image=image.crop(touxiang)
    #     image.save('tmp101.png')
    # else:
        #=======根据位置判断. jietu 
    forqietu=0
    if 1: # 按照名字的切割方案不准.
        if forqietu:#如果高精度的对应版本找到了名字.
            #print(1)
            forqietu
            namexini=[forqietu[1][0],forqietu[0][0],forqietu[0][2]]
            zikuandu=(namexini[2][0]-namexini[1][0])/(len(namexini[0])-2)
            dashuzigaodu
            left=namexini[1][0]-1.5*zikuandu
            right=namexini[1][0]+12.4*zikuandu
            up=namexini[1][1]+1.05*dashuzigaodu
            down=namexini[1][1]+12.05*dashuzigaodu

            # image=image.crop([left,up,right,down])
            import time
            # image.save(str(time.time()*10000000000000)+'.png')






        elif namexini:
            namexini
            zikuandu=(namexini[2][0]-namexini[1][0])/(len(namexini[0])-2)
            dashuzigaodu
            left=namexini[1][0]-0.5*zikuandu
            right=namexini[1][0]+11.4*zikuandu
            up=namexini[1][1]+1.05*dashuzigaodu
            down=namexini[1][1]+13.05*dashuzigaodu

            # image=image.crop([left,up,right,down])
            import time
            # image.save(str(time.time()*10000000000000)+'.png')

        else:
            #print('无法切图')
            pass
    if namexini:
#============计算left


        left=namexini[1][0]




    #==============计算right
        xuliehao=0
        for i in result:
            if out[0] in i[1][0]:
                xuliehao=i
        #print(1)

        namexini
        zikuandu=(xuliehao[0][2][0]-xuliehao[0][0][0])/(len(xuliehao[1][0])-1)
        zigao=(xuliehao[0][2][1]-xuliehao[0][0][1])











        # left=xuliehao[0][2][0]-17.6*zikuandu



        right=xuliehao[0][2][0]-3.3*zikuandu #==========这个准确.


        up=xuliehao[0][2][1]+3.25*zigao
        down=xuliehao[0][2][1]+19.2*zigao
        #print('切割坐标',[left,up,right,down])
        # url='tmp99.png'
        # image = Image.open(url).convert("RGB")
        # image=image.crop([left,up,right,down])
        def cv2_crop(im, box):
            '''cv2实现类似PIL的裁剪

            :param im: 加载好的图像
            :param box: 裁剪的矩形，元组(left, upper, right, lower).
            '''
            return im.copy()[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
        image=cv2_crop(origin_rotaed_img,[left,up,right,down])
        import time
        cv2.imwrite(str(time.time()*10000000000000)+'.png',image)
        xuliehao
        #print()
     












# for i in [ 
#     # 'data/OIP-C.jpg',
#  'data/20230608/1-1.jpg'
    
    
#     ]:
#     chuli(i)




#==========识别后背.






#####=开始做背面的ocr
import cv2
import numpy as np


import cv2 as cv
path=beimian
# path='data/rot.png'

#==========先用图像自带的修正:

#==========图像自动修正.
def  imgRotation(pathtoimg):
        #print('旋转修复的图片是',pathtoimg)
        #图片自动旋正
        from PIL import Image
        img = Image.open(pathtoimg)
        # new_img=cv2.imread(path)
        new_img=cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        if hasattr(img, '_getexif') and img._getexif() != None:
            # 获取exif信息
            dict_exif = img._getexif()
            if 274 in dict_exif:
                if dict_exif[274] == 3:
                    #顺时针180
           
                    new_img=cv2.rotate(new_img,cv2.ROTATE_180)
            
                elif dict_exif[274] == 6:
                    #顺时针90°

                    new_img=cv2.rotate(new_img,cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif dict_exif[274] == 8:
                    #逆时针90°

      
                    new_img=cv2.rotate(new_img,cv2.ROTATE_90_CLOCKWISE)
        return new_img









image=imgRotation(path)

origin_rotaed_img=image








result = ocr.ocr(origin_rotaed_img, cls=True)
# for line in result:
    #print(line)
result=result[0]
result.sort(key=lambda x: x[0][0][1])
#print(result)

out={}
out[result[0][1][0]]=result[1][1][0]
out[result[2][1][0]]=result[3][1][0]
out[result[4][1][0]]=result[5][1][0]
out[result[6][1][0]]=result[7][1][0]

resultbeimian=result
result=result[8:]
qiege=0
for dex,i in enumerate(result):
    if '<<' in i and "ID" in i:
        qiege=dex
result=result[qiege:]
codeee=[i[1][0] for i in result]
beimianmingzizhixindu=result[-1][1][1]
tmp=''.join([i[1][0] for i in result])
#print(tmp)
out['code']=tmp

#print('最后的out',out)









#========根据置信度整合.前后.
result2
outbeimian=out
for j in outbeimian:
    if j=='DISTRICT':
            zhixindu=result[1][1][1]
            for kkk in result2:
                if kkk[0]==out_zhengmian[5]:

                  zhixindu_qianmian=kkk[-1]
            if zhixindu>zhixindu_qianmian:
                out_zhengmian[5]=outbeimian[j]
    
    




#print(1)
#=======
# out['code']
# beimianmingzizhixindu

for kkk in result2:
    if kkk[0]==out_zhengmian[2]:

        zhixindu_qianmian=kkk[-1]
if beimianmingzizhixindu+0.1>zhixindu_qianmian:
        out_zhengmian[2]=[i[1][0] for i in result][-1].replace('<',' ').strip()
print(out_zhengmian,out['code'])
print(time.time()-a9999999999999)

#print('头像保存在最大时间措的png文件.')













