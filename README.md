paddle ocr :   https://gitee.com/paddlepaddle/PaddleOCR/blob/release/2.6/doc/doc_en/inference_args_en.md









#项目规范, 1.py----------->11.py都是主文件, 编号变大表示版本越新.方便回滚找回之前代码, 并且这种数字文件不能当做库包,只能当main函数.
#测试代码  99.py--------->数值更大的文件. 方便跟主文件区分.
正面11.py
背面20.py
原件复印件检测: 30.py


#=====判断图片清晰度:
# 先ocr, 结果看8个结果.如果出不来,或者数字部分有缺少.
# paddleocr, 每半秒拍照, 识别一次,如果出来了,就停. 不出来就一直让你纠正位置.




#=======仿射调整:
https://blog.csdn.net/weixin_39907311/article/details/111393437

https://blog.51cto.com/u_15790101/5677668
https://zhuanlan.zhihu.com/p/520739858

https://pyimagesearch.com/2017/07/17/credit-card-ocr-with-opencv-and-python/#download-the-code

# mtcnn-pytorch
# Descriptions in chinese
  https://blog.csdn.net/Sierkinhane/article/details/83308658

# results:

![](https://github.com/Sierkinhane/mtcnn-pytorch/blob/master/results/r_1.jpg)
![](https://github.com/Sierkinhane/mtcnn-pytorch/blob/master/results/r_2.jpg)
![](https://github.com/Sierkinhane/mtcnn-pytorch/blob/master/results/r_3.jpg)
![](https://github.com/Sierkinhane/mtcnn-pytorch/blob/master/results/r_4.jpg)



# Test an image
  * run > python mtcnn_test.py
 
# Training data prepraring
  * download [WIDER FACE](https://pan.baidu.com/s/1sJTO7TcQ2576RUqR_IIhbQ) (passcode:lsl3) face detection data then store it into ./data_set/face_detection
    * run > python ./anno_store/tool/format/transform.py change .mat(wider_face_train.mat) into .txt(anno_train.txt)
  * download [CNN_FacePoint](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm) face detection and landmark data then store it into ./data_set/face_landmark

# Training
  * preparing data for P-Net
    * run > python mtcnn/data_preprocessing/gen_Pnet_train_data.py
    * run > python mtcnn/data_preprocessing/assemble_pnet_imglist.py
  * train P-Net
    * run > python mtcnn/train_net/train_p_net.py
    
  * preparing data for R-Net
    * run > python mtcnn/data_preprocessing/gen_Rnet_train_data.py (maybe you should change the pnet model path)
    * run > python mtcnn/data_preprocessing/assemble_rnet_imglist.py
  * train R-Net
    * run > python mtcnn/train_net/train_r_net.py
  
  * preparing data for O-Net
    * run > python mtcnn/data_preprocessing/gen_Onet_train_data.py
    * run > python mtcnn/data_preprocessing/gen_landmark_48.py
    * run > python mtcnn/data_preprocessing/assemble_onet_imglist.py
  * train O-Net
    * run > python mtcnn/train_net/train_o_net.py
    
 # Citation
   [DFace](https://github.com/kuaikuaikim/DFace)
# ocr_private
# ocr_backup
