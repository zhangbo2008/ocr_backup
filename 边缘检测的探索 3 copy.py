
#======从mask 变到 图像的四角定位.

import cv2  #######边缘检测.
import argparse
import imutils
import cv2

import numpy as np




import argparse
import imutils
import cv2

import cv2
import numpy as np
 
#写一个方法用于展示图像，展示后按任意键继续下行代码
def cv_show(title,img):
    cv2.imshow(title,img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    return
 
img = cv2.imread('debug999.png').astype(np.uint8)*1 

# cv2.imwrite('img.png',img)











#就是上一章的内容，具体就是会输出一个轮廓图像并返回一个轮廓数据
def draw_contour(img,color,width):
    import numpy as np
    kernel = np.ones((1, 5), np.uint8)
    if len(img.shape)>2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#转换灰度图
    else:
        gray=img
    cv2.imwrite('gray.png',gray)
    ret, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
    binary2 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=22) # 二值化. 
    #模糊化.

    



    binary = cv2.blur(img,(5,5))
    binary = cv2.blur(binary,(5,5))
    binary = cv2.blur(binary,(5,5))
    binary = cv2.boxFilter(binary,-1,(5,5), normalize=True )
    cv2.imwrite('binary998.png',binary)
    if 1:                     
    # 边缘检测, Sobel算子大小为3
        edges = cv2.Canny(binary, 100, 200, apertureSize=3)
        # 霍夫曼直线检测
        cv2.imwrite('binary997.png',edges )

        gao=edges.shape[0]
        chang=edges.shape[1]




        lines = cv2.HoughLinesP(edges, 1, 1*np.pi / 180, int((gao+chang)/40), minLineLength=(gao+chang)/20, maxLineGap=(gao+chang)/20)



#setp1: 现在各个直线.进行分成4类. 然后再拟合4个直线.!!!!!!!!!!!!!!!
        #=======首先算出 每个直线的旋转角度.
        #====直接算斜率
        # xielv=[]
        # for i in lines:
        #     xielv.append('inf' if not i[0][3]-i[0][1] else (i[0][3]-i[0][1])/(i[0][2]-i[0][0]))
        #     print()

        print()
        import math
        #=========输入一个直线, 计算他跟x轴的夹角.
        def jiajiao(line):   # 4点决定一个线
                if (line[2]-line[0]) :
                    a=math.atan((line[3]-line[1])/(line[2]-line[0]))/math.pi*180 
                    # if a<0:
                    #     return 180+a
                    return a
                else:
                    return 90
        # a=jiajiao([0,0,-1,1])
        a=[jiajiao([i[0][0],i[0][1],i[0][2],i[0][3]]) for i in lines]
        jiajiaobaocun=a

        import matplotlib.pyplot as plt
        import numpy as np
        # import libraries
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        a=[round(i) for i in a]
        print(a)
        plt.hist(a, density=False)
        plt.savefig('aaaa.png')
        print()
        # 按照相差上下10度来分类.

        #============下面按照上下15度进行分类.因为每一个类别一定有一个中间轴.中间轴一定是这些店里面的值.我们来存索引.
        # yuzhi=15
        if a:#
                pass
                #根据我们身份证的理解.
                # 我们要的2个中州一定是距离最大的.一般是0和90度.
                juli=0#先算出距离最大值.
                for dex,i in enumerate(a):
                    for dex2,j in enumerate(a):
                        tmp=min(abs(i-j),abs(180+i-j),abs(180-i+j))
                        if tmp>juli:
                            baocun=dex,dex2
                            juli=tmp
                print(baocun,juli)
                yuzhi=juli/3
        else:
            print('没有任何一个直线,所以算法不进行后续边界识别')
        if baocun:
            j=a[baocun[0]]
            list1=[dex for dex,i in enumerate(a) if min(abs(i-j),abs(180+i-j),abs(180-i+j))<yuzhi]
            j=a[baocun[1]]
            list2=[dex for dex,i in enumerate(a) if min(abs(i-j),abs(180+i-j),abs(180-i+j))<yuzhi]
        print('打印两组直线角度阵营',list1,list2)
        #=========分别算投影, 去掉一个方向分量之后我们进行第二次细分这2住店.这样就得到了4个边的阵营.


        
        
        zhixianfenzu=[]
        a1=list1
        a2=list2
        for aaa in [a1,a2]:
            #======算出每个阵营的投影直线.
            #==先算每个阵营的中心直线
            zhenying=aaa
            zhenyingjiaodu=[a[i] for i in list1]
            zhenyingzhixianjiaodu=sum(zhenyingjiaodu)/len(zhenyingjiaodu)
            print(zhenyingzhixianjiaodu)
            chuizhijiaodu=zhenyingzhixianjiaodu+90


            a=math.tan(chuizhijiaodu/180*math.pi)
            xiangliang=(1,a*1)
            list1zhongdian=[[(lines[i][0][0]+lines[i][0][2])/2,(lines[i][0][1]+lines[i][0][3])/2] for i in list1]
            touying=[(i[0]*xiangliang[0]+i[1]*xiangliang[1])/math.sqrt(xiangliang[0]**2+xiangliang[1]**2) for i in list1zhongdian]




            #=========继续用间隔来分类
            juli=0#先算出距离最大值.
            for dex,i in enumerate(touying):
                for dex2,j in enumerate(touying):
                    tmp=abs(i-j)
                    if tmp>juli:
                        baocun=dex,dex2
                        juli=tmp
            print(baocun,juli)
            yuzhi=juli/3
            print(a)
            a=touying
            if baocun:
                j=a[baocun[0]]
                list1=[dex for dex,i in enumerate(a) if abs(i-j)<yuzhi]
                j=a[baocun[1]]
                list2=[dex for dex,i in enumerate(a) if abs(i-j)<yuzhi]

            print(1)
            list1inalldex=[zhenying[i] for i in list1]
            list2inalldex=[zhenying[i] for i in list2]
            zhixianfenzu.append(list1inalldex)
            zhixianfenzu.append(list2inalldex)
            print()
        print()
        zhixianfenzu# 里面有4个数组, 每个数组表示一个直线族. 
        #=========下面把每组的直线拟合成一条直线
        #==================
        all_four_line=[]
        if len(zhixianfenzu)==4:
            pass
        #没太好思路, 就平均数吧
            for i in zhixianfenzu:
                tmpzhixian= np.squeeze(lines[i], axis = 1)
                tmpjiajiao=np.array(jiajiaobaocun)[i].mean()
                tmpzhongxindian=np.array([(tmpzhixian[:,0]+tmpzhixian[:,2])/2,(tmpzhixian[:,1]+tmpzhixian[:,3])/2]).T


                tmpzhongdian2=tmpzhixian.mean(axis=0)
                tmpzhongdian2=(tmpzhongdian2[0]+tmpzhongdian2[2])/2,(tmpzhongdian2[1]+tmpzhongdian2[3])/2
                print(1)
                all_four_line.append([tmpzhongdian2,tmpjiajiao])
        #=======转化为双点是.
        all_four_line2=[]
        for i in all_four_line:
                dian=i[0]
                jiaodu=i[1]
                a=math.tan(jiaodu/180*math.pi)
                all_four_line2.append([dian[0],dian[1],dian[0]+1,dian[1]+a])
        all_four_line=all_four_line2
        #===========计算交点
        def cross_point(line1, line2):  # 计算交点函数
            #是否存在交点
            point_is_exist=False
            x=0
            y=0
            x1 = line1[0]  # 取四点坐标
            y1 = line1[1]
            x2 = line1[2]
            y2 = line1[3]

            x3 = line2[0]
            y3 = line2[1]
            x4 = line2[2]
            y4 = line2[3]

            if (x2 - x1) == 0:
                k1 = None
            else:
                k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
                b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键

            if (x4 - x3) == 0:  # L2直线斜率不存在操作
                k2 = None
                b2 = 0
            else:
                k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
                b2 = y3 * 1.0 - x3 * k2 * 1.0

            if k1 is None:
                if not k2 is None:
                    x = x1
                    y = k2 * x1 + b2
                    point_is_exist=True
            elif k2 is None:
                x=x3
                y=k1*x3+b1
            elif not k2==k1:
                x = (b2 - b1) * 1.0 / (k1 - k2)
                y = k1 * x * 1.0 + b1 * 1.0
                point_is_exist=True
            return point_is_exist,[x, y]
        #=======算每一个直线跟其他直线的交点
        #diyige :
        all3=[]
        tmp=all_four_line[0]
        tmp2=all_four_line[1]
        jiaodian1=cross_point(tmp,tmp2)
        tmp=all_four_line[0]
        tmp2=all_four_line[2]
        jiaodian2=cross_point(tmp,tmp2)
        tmp=all_four_line[0]
        tmp2=all_four_line[3]
        jiaodian3=cross_point(tmp,tmp2)
        all2=[]
        if jiaodian1[0]:
                all2.append(jiaodian1[1])
        if jiaodian2[0]:
                all2.append(jiaodian2[1])   
        if jiaodian3[0]:
                all2.append(jiaodian3[1])
        all2.sort(key=lambda x:abs(x[0])+abs(x[1]) )
        all2=all2[:2]
        all3+=all2


        tmp=all_four_line[1]
        tmp2=all_four_line[0]
        jiaodian1=cross_point(tmp,tmp2)
        tmp=all_four_line[1]
        tmp2=all_four_line[2]
        jiaodian2=cross_point(tmp,tmp2)
        tmp=all_four_line[1]
        tmp2=all_four_line[3]
        jiaodian3=cross_point(tmp,tmp2)
        all2=[]
        if jiaodian1[0]:
                all2.append(jiaodian1[1])
        if jiaodian2[0]:
                all2.append(jiaodian2[1])   
        if jiaodian3[0]:
                all2.append(jiaodian3[1])
        all2.sort(key=lambda x:abs(x[0])+abs(x[1]) )
        all2=all2[:2]
        all3+=all2
        print(1)

        #=====================check!!!!!!!!!!
        for line in lines:
        # 获取坐标 
            x1, y1, x2, y2 = line[0]
            # cv2.line(binary, (x1, y1), (x2, y2), (0, 255, 255), thickness=25)
        cv2.imwrite('binary999.png',binary)


        #==============画点.
        if 0: # 原图画点.
            binary2=binary
            binary=cv2.imread('debug99k.png')
            binary=cv2.resize(binary,binary2.shape[:2][::-1])
            cv2.imwrite('debug1003.png',binary)
        binary=gray
        for i in all3:
            binary=cv2.circle(binary,(int(i[0]),int(i[1])),40,(255,255,255),2)
        print(all3,'所有的定位点!!!!!!!!!!!!最终的')
        cv2.imwrite('debug1002.png',binary)
        #=======因为平行肯定有一个线超长.
        # contours,hierarchy = cv2.findContours(binary2,cv2.RETR_CCOMP  ,cv2.CHAIN_APPROX_SIMPLE)
        # tuxingzhouchang=cv2.arcLength(contours[0], True)
        # print(1)
        #paixu jike





        # lines = cv2.HoughLines(edges,1,np.pi/180,100)
      

    dsfadsfads
        
    contours,hierarchy = cv2.findContours(binary,cv2.RETR_CCOMP  ,cv2.CHAIN_APPROX_SIMPLE)
    cv2.arcLength(contour, True)



    contours = sorted(contours, key = cv2.contourArea, reverse = True)[0]
    copy = img.copy()
    resoult = cv2.drawContours(copy,contours[:],-1,color,width)
    cv2.imwrite('resoult.png',resoult)
    return contours
 
 
contour = draw_contour(img,(0,0,255),2)
print(type(contour))    
if 0:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    print(type(approx),approx.shape)

    res = cv2.drawContours(img.copy(),[approx],-1,(0,255,255),2)


    ttt=cv2.imread('data/tq2.png')
    ttt=cv2.resize(ttt,(1280,845))
    res2=cv2.drawContours(ttt.copy(),[approx],-1,(0,255,255),2)
    cv2.imwrite('res.png',res) #打印最后的多边形矿.
    cv2.imwrite('res2.png',res2) #打印最后的多边形矿.

    print('对应的原始图片是','data/tq2.png')


