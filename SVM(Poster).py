import pandas
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn import svm
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np 

features = pandas.read_excel('dataset.xlsx')
features.head()
#print ('We have {} days of data with {} variables'.format(*features.shape))
print('The shape of our features is:', features.shape)
#features.describe()

import tkinter as tk

master = tk.Tk()
master.geometry("450x550")
master.title("Air qulity")
tk.Label(master, text="AQI",fg='blue').grid(row=0)
tk.Label(master, text="PM2.5",fg ='blue').grid(row=1)
tk.Label(master, text="PM2.5_24h",fg ='blue').grid(row=2)
tk.Label(master, text="PM10",fg ='blue').grid(row=3)
tk.Label(master, text="PM10_24h",fg ='blue').grid(row=4)
tk.Label(master, text="SO2",fg ='blue').grid(row=5)
tk.Label(master, text="SO2_24h",fg ='blue').grid(row=6)
tk.Label(master, text="NO2",fg ='blue').grid(row=7)
tk.Label(master, text="NO2_24h",fg ='blue').grid(row=8)
tk.Label(master, text="O3",fg ='blue').grid(row=9)
tk.Label(master, text="O3_24h",fg ='blue').grid(row=10)
tk.Label(master, text="O3_8h",fg ='blue').grid(row=11)
tk.Label(master, text="O3_8h_24h",fg ='blue').grid(row=12)
tk.Label(master, text="CO",fg ='blue').grid(row=13)
tk.Label(master, text="CO_24h",fg ='blue').grid(row=14)
tk.Label(master, text="Latitude",fg ='blue').grid(row=15)
tk.Label(master, text="Longtitude",fg ='blue').grid(row=16)
#tk.Label(master, text="Latitude2",fg ='blue').grid(row=17)
#tk.Label(master, text="Longtitude2",fg ='blue').grid(row=18)
#tk.Label(master, text="Latitude3",fg ='blue').grid(row=19)
#tk.Label(master, text="Longtitude3",fg ='blue').grid(row=20)
tk.Label(master, text="hour",fg ='blue').grid(row=21)

#variables
AQI = tk.StringVar()
PM2_5 = tk.StringVar()
PM2_5_24h = tk.StringVar()
PM10 = tk.StringVar()
PM10_24h  = tk.StringVar()
SO2 = tk.StringVar()
SO2_24h = tk.StringVar()
NO2 = tk.StringVar()
NO2_24h = tk.StringVar()
O3 = tk.StringVar()
O3_24h = tk.StringVar()
O3_8h = tk.StringVar()
O3_8h_24h = tk.StringVar()
CO = tk.StringVar()
CO_24h  = tk.StringVar()
Latitude  = tk.StringVar()
Longtitude =  tk.StringVar()
#Latitude2  = tk.StringVar()
#Longtitude2 =  tk.StringVar()
#Latitude3  = tk.StringVar()
#Longtitude3 =  tk.StringVar()
hour  = tk.StringVar()

e1 = tk.Entry(master,textvariable= AQI)
e2 = tk.Entry(master,textvariable= PM2_5)
e3 = tk.Entry(master,textvariable= PM2_5_24h)
e4 = tk.Entry(master,textvariable= PM10)
e5 = tk.Entry(master,textvariable= PM10_24h)
e6 = tk.Entry(master,textvariable= SO2)
e7 = tk.Entry(master,textvariable = SO2_24h)
e8 = tk.Entry(master,textvariable = NO2)
e9 = tk.Entry(master,textvariable = NO2_24h)
e10 = tk.Entry(master,textvariable = O3)
e11 = tk.Entry(master,textvariable = O3_24h)
e12 = tk.Entry(master,textvariable = O3_8h)
e13 = tk.Entry(master,textvariable = O3_8h_24h)
e14 = tk.Entry(master,textvariable = CO)
e15 = tk.Entry(master,textvariable = CO_24h)
e16 = tk.Entry(master,textvariable = Latitude)
e17 = tk.Entry(master,textvariable = Longtitude)
#e18 = tk.Entry(master,textvariable = Latitude2)
#e19 = tk.Entry(master,textvariable = Longtitude2)
#e20 = tk.Entry(master,textvariable = Latitude3)
#e21 = tk.Entry(master,textvariable = Longtitude3)
e22 = tk.Entry(master,textvariable = hour)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)
e5.grid(row=4, column=1)
e6.grid(row=5, column=1)
e7.grid(row=6, column=1)
e8.grid(row=7, column=1)
e9.grid(row=8, column=1)
e10.grid(row=9, column=1)
e11.grid(row=10, column=1)
e12.grid(row=11, column=1)
e13.grid(row=12, column=1)
e14.grid(row=13, column=1)
e15.grid(row=14, column=1)
e16.grid(row=15, column=1)
e17.grid(row=16, column=1)
#e18.grid(row=17, column=1)
#e19.grid(row=18, column=1)
#e20.grid(row=19, column=1)
#e21.grid(row=20, column=1)
e22.grid(row=21, column=1)
y = features["Class"]
x = features.drop(["Class"],axis=1)



train_x = x[:120000]
train_y = y[:120000]
test_x = x[120000:]
test_y = y[120000:]


clf = svm.SVC()
model = clf.fit(train_x, train_y)
result = model.predict(test_x)
print(result)
print(test_y)
print(model.score(test_x,test_y))

cm = confusion_matrix(test_y,result)
print(cm)
accuracy = accuracy_score(test_y,result)
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
