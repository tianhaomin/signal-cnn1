# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:22:24 2018

@author: Administrator
"""
##########################################补充实验
import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn
import os
from pandas import DataFrame
import tensorflow as tf
import random
from sklearn import preprocessing 
train = np.load("F://tmp//data//raw data//train.npy")
label1=np.load("F://tmp//data//raw data//label_train.npy")
test=np.load("F://tmp//data//raw data//test.npy")
label1_test=np.load("F://tmp//data//raw data//label_test.npy")
# SVM 
from sklearn import svm
clf = svm.SVC()
x = []
for i in train:
    x.append(list(i))
y = []
for i in label1:
    tmp = list(i)
    val = tmp.index(1)
    y.append(val)
clf.fit(x, y)
test1 = []
for i in test:
    test1.append(list(i))
result = list(clf.predict(test1))
label = []
for i in label1_test:
    tmp = list(i)
    val = tmp.index(1)
    label.append(val)
count = 0
for i in range(len(result)):
    if result[i] == label[i]:
        count += 1
acc1 = count / 124.0
# Gaussian Navie Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
clf2 = gnb.fit(x, y)
result2 = clf2.predict(test1)
count = 0
for i in range(len(result2)):
    if result2[i] == label[i]:
        count += 1
acc2 = count / 155.0
# 神经网络
from sklearn.neural_network import MLPClassifier
clf3 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(128, 128), random_state=1)
clf3.fit(train,label1)
result3 = clf3.predict(test)
count = 0
for i in range(len(result3)):
    tmp = list(result3[i])
    val = tmp.index(1)
    if val == label[i]:
        count += 1
acc3 = count / 155.0
#KNN
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x, y) 
result4 = neigh.predict(test1)
count = 0
for i in range(len(result4)):
    if result4[i] == label[i]:
        count += 1
acc4 = count / 155.0



#迁移效果对比
a = os.listdir("F:/project/Yin/spectrum-data")
def read_data(file_num,start_frq,end_frq):
    z1 = DataFrame({})
    for i in range(file_num):
        df = pandas.read_table("F:/project/Yin/spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(start_frq,end_frq)]
        z1 = pandas.concat([z1,df1['E']])
        z2 = z1.values
        z3 = np.array([z2])
    return z3
#############cdma_down#########2
data_cdma_down = read_data(850,870,880).reshape(850,401)
##############egsm_down#########4
data_egsm_down = read_data(850,930,940).reshape(850,401)
data_wcdma_down = read_data(850,2135,2145).reshape(850,401)
###############4G################6
data_lte_down = read_data(850,1850,1860).reshape(850,401)
#data_evdo_down = read_data(850,1920,1930).reshape(850,401)
data_dcs_down = read_data(850,1900,1910).reshape(850,401)
#
data_cdma_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_cdma_down] )
data_egsm_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_egsm_down] )
data_wcdma_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_wcdma_down] )
data_lte_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_lte_down] )
#data_evdo_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_evdo_down] )
data_dcs_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_dcs_down] )
#
train_data = [data_cdma_down,data_egsm_down,data_wcdma_down,data_lte_down,data_dcs_down]
train = train_data[0]
for i in range(len(train_data)-1):
    train = np.vstack((train,train_data[i+1]))
enc = preprocessing.OneHotEncoder()  
enc.fit([[1],[2],[3],[4],[5]])  
array1_1 = enc.transform([[1]]*850).toarray()  
array1_2 = enc.transform([[2]]*850).toarray()  
array1_3 = enc.transform([[3]]*850).toarray()  
array1_4 = enc.transform([[4]]*850).toarray()  
array1_5 = enc.transform([[5]]*850).toarray()  
label1 = np.vstack((array1_1,array1_2,array1_3,array1_4,array1_5))

############cdma_down#########2
test_cdma_down = read_data(31,870,880).reshape(31,401)
#############egsm_down#########4
test_egsm_down = read_data(31,930,940).reshape(31,401)
##########lte下行#########
test_lte_down = read_data(31,1850,1860).reshape(31,401)
#########wcdma下行###########
test_wcdma_down = read_data(31,2135,2145).reshape(31,401)
#test_evdo_down = read_data(31,1920,1930).reshape(31,401)
test_dcs_down = read_data(31,1900,1910).reshape(31,401)

test_cdma_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_cdma_down] )
test_egsm_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_egsm_down] )
test_lte_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_lte_down] )
test_wcdma_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_wcdma_down] )
#test_evdo_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_evdo_down] )
test_dcs_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_dcs_down] )
##############
test_data = [test_cdma_down,test_egsm_down,test_wcdma_down,test_lte_down,test_dcs_down]

test = test_data[0]
for i in range(len(test_data)-1):
    test = np.vstack((test,test_data[i+1]))

array1_1_test = enc.transform([[1]]*31).toarray()
array1_2_test = enc.transform([[2]]*31).toarray()
array1_3_test = enc.transform([[3]]*31).toarray()
array1_4_test = enc.transform([[4]]*31).toarray()
array1_5_test = enc.transform([[5]]*31).toarray()
label1_test = np.vstack((array1_1_test,array1_2_test,array1_3_test,array1_4_test,array1_5_test))
label1 = label1.astype(np.float32)
label1_test = label1_test.astype(np.float32)   
train=train.astype(np.float32)
test = test.astype(np.float32)
#打乱顺序
idx=random.sample(range(4250),4250) 
idx1 = random.sample(range(155),155)
label1_test = label1_test[idx1]
test = test[idx1]
label1 = label1[idx]
train = train[idx]   
  
