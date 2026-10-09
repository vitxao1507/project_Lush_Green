import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import machine
import pandas as pd
from data_loader import load_data,creat_data
import tensorflow as tf
import gc
n=10
for i in range(n):
    print(f"Lần train thứ {i+1}")
    tf.keras.backend.clear_session()
    gc.collect()
    model=machine.lush_green_9()
    data_train_path="data_set/data_train_2000"
    validation_data_path="data_set/validation"
    x_data,y_data=load_data(data_train_path)
    x_val,y_val=load_data(validation_data_path)
    data_set=creat_data(x_data,y_data,shuffle=True)
    validation_data_set=creat_data(x_val,y_val,shuffle=False)
    
    checkpoint=tf.keras.callbacks.ModelCheckpoint(
    f"model_saved/semi_2000/best_lush_green_semi_2000-run_{i+1}.keras",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )
    history=model.fit(
    data_set,
        validation_data=validation_data_set,
        epochs=100,
        callbacks=[checkpoint],
        shuffle=False
    )
    os.makedirs("history", exist_ok=True)
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(
        f"history/semi_2000/history train/history_semi_2000-run_{i+1}.csv",
        index=False,
        encoding="utf-8"
    )
    print(y_data.shape)
    print(y_val.shape)
    print("Lưu thành công!")
#loss dataset
#loss vali