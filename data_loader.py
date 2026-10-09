import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pre
import tensorflow as tf
#khởi chạy data từ dataset
def load_data(data_path):
    labels=[]
    images=[]
    labels_dict={
        "healthy":0,
        "curled":1,
        "spot":2,
        "yellowing":3,
    }
    for disease in labels_dict:
        label=labels_dict[disease]
        folder=os.path.join(
            data_path,
            disease
        )
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img=pre.open_pic(img_path)
            img=pre.preprocessing(img)
            if img is None:
                continue
            labels.append(label)
            images.append(img)
    x=np.array(images)#=(224,224,3)
    y=np.array(labels)#=(200,)
    return x,y
#tạo dataset cho mô hình
def creat_data(x,y,shuffle=True):
    dataset=tf.data.Dataset.from_tensor_slices((x,y))
    if shuffle:
        dataset=dataset.shuffle(len(x))
    dataset=dataset.batch(32)
#khi GPU đang train thì CPU chuẩn bị nguyên liệu (tối ưu hiệu năng)
    dataset=dataset.prefetch(tf.data.AUTOTUNE)
    return dataset