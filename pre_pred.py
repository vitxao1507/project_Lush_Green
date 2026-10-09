# mở ảnh , chuyển thành RGB rồi mảng hóa
import numpy as np
from PIL import Image
import cv2
# mở ảnh
def open_pic(path):
    img=Image.open(path)
    img=img.convert("RGB")
    arr=np.array(img)
    img=cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)
    img_final=cv2.resize(img,(224,224))
    img_final=cv2.cvtColor(img_final,cv2.COLOR_BGR2RGB)
    arr_train=img_final.astype(np.float32)/255.0
    return arr_train
# (224,224,3)