#ceshi paddleocr # python 3.9.1  # https://zhuanlan.zhihu.com/p/380142530
# pip3 install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
# -i https://mirror.baidu.com/pypi/simple
from paddleocr import PaddleOCR, draw_ocr
# use_angle_cls参数用于确定是否使用角度分类模型，即是否识别垂直方向的文字。
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False,lang='en')
img_path = r'tt.jpg'
result = ocr.ocr(img_path, cls=True)
for line in result:
    print(line)