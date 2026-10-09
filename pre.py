# mở ảnh , chuyển thành RGB rồi mảng hóa
import numpy as np
from PIL import Image
import cv2

# mở ảnh
def open_pic(path):
    img=Image.open(path)
    img=img.convert("RGB")
    arr=np.array(img)
    return arr
#tiền xử lí
def preprocessing(a):
    arr= a
    img=cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower=np.array([20,25,25])
    upper=np.array([85,255,255])
    mask=cv2.inRange(hsv,lower,upper)
    contours,hierarchy=cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours)==0:
        return None
    else:
        largest=max(contours,key=cv2.contourArea)
        x,y,w,h=cv2.boundingRect(largest)
    img_final=img[y:y+h,x:x+w]
    img_final=cv2.resize(img_final,(224,224))
    img_final=cv2.cvtColor(img_final,cv2.COLOR_BGR2RGB)
    arr_train=img_final.astype(np.float32)/255.0
    return arr_train
# (224,224,3)