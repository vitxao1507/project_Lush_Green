import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import pre_pred
import pre
import numpy as np
model = tf.keras.models.load_model("model_saved/run_500_final/best_lush_green_500-run_3.keras")
img=pre.open_pic("data_set/4.png")
img=pre.preprocessing(img)
img = np.expand_dims(img, axis=0)
#(1,224,224,3)
prediction = model.predict(img)
print(prediction)
for i, p in enumerate(prediction):
    print(f"Ảnh {i+1}")
    print(f"Healthy    : {p[0]*100:.2f}%")
    print(f"Curled     : {p[1]*100:.2f}%")
    print(f"Spot       : {p[2]*100:.2f}%")
    print(f"Yellowing  : {p[3]*100:.2f}%")
    print(f"Tổng       : {np.sum(p)*100:.2f}%")
#[0.1,0.4.,0.3,0.2]