#=======paddleocr进行判断图片是否放的正确. 

from paddleocr import PaddleOCR, draw_ocr
    # use_angle_cls参数用于确定是否使用角度分类模型，即是否识别垂直方向的文字。

if 1:
    ocr = PaddleOCR(use_angle_cls=False, use_gpu=False,

    lang='en',

    # det_model_dir="PaddleOCR/inference/ch_ppocr_server_v2.0_det_infer"  ,
    # rec_model_dir="PaddleOCR/inference/ch_ppocr_server_v2.0_rec_infer"  ,
    # cls_model_dir="PaddleOCR/inference/ch_ppocr_mobile_v2.0_cls_infer"  ,

    use_space_char=True,

    
    )

    img_path = r'data/tq.jpg'


    import time
    aaa=time.time()
    result = ocr.ocr(img_path, cls=True)[0]
    def aaa1(result):
        for line in result:
            if 'JAMHURIYA' in line[1][0]:
                return True
        return False
    print(aaa1(result))
    print(time.time()-aaa)
    