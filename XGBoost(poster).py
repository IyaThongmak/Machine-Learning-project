import xgboost as xgb
import pandas

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix,accuracy_score

features = pandas.read_excel('dataset.xlsx')
features.head()

from sklearn.model_selection import train_test_split


list_y = []
y = features["Class"]
value, count = np.unique(y, return_counts=True)
for i in y:
    if(i=="Good"):
        list_y.append(0)
    if(i=="Moderate"):
        list_y.append(1)
    if(i=="Unhealthy for Sensitive Groups"):
        list_y.append(2)
    if(i=="Unhealthy "):
        list_y.append(3)
    if(i=="Very Unhealthy"):
        list_y.append(4)
    if(i=="Hazardous"):
        list_y.append(5)
        
x = features.drop(["Class"],axis=1)



train_features, test_features, train_labels, test_labels = train_test_split(x, y,test_size = 0.2, random_state = 42)

clf = xgb.XGBClassifier().fit(train_features,train_labels)
result = clf.predict(test_features)

cm = confusion_matrix(test_labels,result)
print(cm)
accuracy = accuracy_score(test_labels,result)
print("Accuracy",accuracy)


recall = np.diag(cm) / np.sum(cm, axis = 1)
recall = pandas.DataFrame(recall)
recall = recall.replace(np.nan,0)
recall = np.array(recall)

precision = np.diag(cm) / np.sum(cm, axis = 0)
precision = pandas.DataFrame(precision)
precision = precision.replace(np.nan,0)
precision = np.array(precision)

AVG_precision = sum(precision)/len(precision)
AVG_precision = AVG_precision*100
AVG_precision = AVG_precision[0]

AVG_recall = sum(recall)/len(recall)
AVG_recall = AVG_recall*100
AVG_recall = AVG_recall[0]

AVG_f1 = 2 * ((AVG_precision * AVG_recall)/(AVG_precision + AVG_recall))

print("AVG_precision",AVG_precision)
print("AVG_recall",AVG_recall)
print("AVG_f1",AVG_f1)


cm_with_precision = cm/np.sum(cm, axis = 0)
cm_with_precision = np.nan_to_num(cm_with_precision)

cm_with_recall = (cm.T/np.sum(cm, axis = 1)).T
cm_with_recall = np.nan_to_num(cm_with_recall)
print(cm_with_precision)
print(cm_with_recall)
