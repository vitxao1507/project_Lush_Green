import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import data_loader as dl
import numpy as np
import csv
import pandas as pd

list_acc=[]
list_pre=[]
list_f1=[]
list_loss=[]
n=10

for i in range(n):
    test_path="data_set/test"
    x_test,y_test=dl.load_data(test_path)
    data_test=dl.creat_data(x_test,y_test,shuffle=False)
    y_true = y_test
    model= tf.keras.models.load_model(f"model_saved/semi_2000/best_lush_green_semi_2000-run_{i+1}.keras")
    loss, accuracy = model.evaluate(data_test)
    predict=model.predict(data_test)  
    y_pred=np.argmax(predict, axis=1)

    from sklearn.metrics import precision_score
    precision = precision_score(
        y_true,
        y_pred,
        average="macro"
    )

    from sklearn.metrics import recall_score
    recall = recall_score(
        y_true,
        y_pred,
        average="macro"
    )

    from sklearn.metrics import f1_score
    f1= f1_score(
        y_true,
        y_pred,
        average= "macro"
    )

    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(
        y_true,
        y_pred
    )

    print(f"Loss      : {loss:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"C_matrix  : \n{cm}")

    list_acc.append(accuracy)
    list_f1.append(f1)
    list_pre.append(precision)
    list_loss.append(loss)

    file_path_rs="history/semi_2000/result/Test_result/test.csv"
    with open(file_path_rs,"a",newline="",encoding="utf-8-sig") as g:
        writer=csv.writer(g)
        writer.writerow([
            "Accuracy (%)",
            "Loss (%)",
            "Precision (%)",
            "Recall (%)",
            "F1-score (%)"
        ])

        writer.writerow([
            accuracy * 100,
            loss * 100,
            precision * 100,
            recall * 100,
            f1 *100
        ])

    label_name=["Healthy","Curled","Spot","Yellowing"]
    cm_pd=pd.DataFrame(cm,index=label_name,columns=label_name)
    cm_pd.to_csv(
        f"history/semi_2000/result/Test_matrix/C-matrix {i+1}.csv",
        index=True,
        encoding="utf-8"
    )
    
    os.makedirs("Test_results",exist_ok=True)
    file_path=f"Test_results/semi_test_2000/test_{i+1}.csv"
    with open(file_path,"a",newline="",encoding="utf-8-sig") as f:
        writer=csv.writer(f)
        writer.writerow([
            "index",
            "true_label",
            "pred_label",
            "healthy (%)",
            "curled (%)",
            "spot (%)",
            "yellowing (%)",
            "confidence (%)",
            "correct"
        ])

        for i in range(len(y_pred)):
            true_label=label_name[y_true[i]]
            pred_label=label_name[y_pred[i]]
            confidence= predict[i][y_pred[i]]
            correct= true_label == pred_label
            writer.writerow([
                i,
                true_label,
                pred_label,
                predict[i][0] * 100,
                predict[i][1] * 100,
                predict[i][2] * 100,
                predict[i][3] * 100,
                confidence * 100,
                correct
            ])

        print("Đã lưu kết quả test tại:")
        print(file_path)

mean_acc=np.mean(list_acc)
mean_pre=np.mean(list_pre)
mean_f1=np.mean(list_f1)
mean_loss=np.mean(list_loss)
mean_recall=mean_acc
sd_acc=np.std(list_acc)
max_acc=max(list_acc)
min_acc=min(list_acc)

file_path_rs=f"history/semi_2000/result/Test_result/Mean SD.csv"
with open(file_path_rs,"a",newline="",encoding="utf-8-sig") as h:
    writer=csv.writer(h)
    writer.writerow([
        "Accuracy mean (%)",
        "Loss mean (%)",
        "Precision mean(%)",
        "Recall mean (%)",
        "F1-score mean (%)",
        "SD acc (%)",
        "Min",
        "Max"
    ])

    writer.writerow([
        mean_acc * 100,
        mean_loss * 100,
        mean_pre * 100,
        mean_recall * 100,
        mean_f1 * 100,
        sd_acc * 100 ,
        min_acc * 100,
        max_acc * 100
    ])

print(predict.shape)
print(y_pred.shape)
print(y_true.shape)