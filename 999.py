from paddleocr import PaddleOCR, draw_ocr
import os
import numpy as np
import glob

import multiprocessing
aa=multiprocessing.cpu_count()
#imagefiles=os.listdir(imagepath)
#imagefiles = os.walk(imagepath)
image_files = glob.glob('data/20230608/*.*')
import time
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_dilation=True,use_mp=True,total_process_num=aa   ,


# precision='fp16'
# det_algorithm='DB+',
# det_db_score_mode='slow',
# rec_algorithm='SVTR_LCNet',

)  # need to run only once to download and load model into memory
a=time.time()
image_files=['data/20230608/1-1.jpg']
for image in image_files:
    print(image)


    if 0:
        import cv2
        img=cv2.imread(image)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
        ret, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE) 
        kernel = np.ones((1, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=1) # 二值化.

        cv2.imwrite("img2.png", binary)   
        image="img2.png"





#    img_path = image_path + image
    print('==========>' + image)

    
    result = ocr.ocr(image, cls=True)
    for idx in range(len(result)):
        for line in result[idx] :
            print(line)
print(time.time()-a)