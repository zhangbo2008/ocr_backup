# ========只能识别单行的. 先不用.

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

# load image from the IAM database (actually this model is meant to be used on printed text)
url = 'tmp99.png'
image = Image.open(url).convert("RGB")

d='/mnt/e/trocr-base_printed'
processor = TrOCRProcessor.from_pretrained(d)
# model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained(d)
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
