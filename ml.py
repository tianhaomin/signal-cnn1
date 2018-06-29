# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 17:04:19 2018

@author: Administrator
"""
import numpy as np
from sklearn import svm
#SVM
train = np.load("F://tmp//data//raw data//train.npy")
label1=np.load("F://tmp//data//raw data//label_train.npy")
test=np.load("F://tmp//data//raw data//test.npy")
label1_test=np.load("F://tmp//data//raw data//label_test.npy")
for i in train:
    idx=random.sample(range(200),180) 
    i[idx] = 0

for i in test:
    idx=random.sample(range(200),180) 
    i[idx] = 0
label2 = []  
label2_test = []
for i in label1:
    if i[0] == 1:
        label2.append(1)
    if i[1] == 1:
        label2.append(2)
    if i[2] == 1:
        label2.append(3)
    if i[3] == 1:
        label2.append(4)

for i in label1_test:
    if i[0] == 1:
        label2_test.append(1)
    if i[1] == 1:
        label2_test.append(2)
    if i[2] == 1:
        label2_test.append(3)
    if i[3] == 1:
        label2_test.append(4)
clf = svm.SVC()
clf.fit(train, label2)    
result1=clf.predict(test)
error_num1 = 0
for i in range(124):
    if result1[i] != label2_test[i]:
        error_num1 += 1

acc1 = (124-error_num1)/124.0

#bayes
from sklearn.naive_bayes import GaussianNB
train = np.load("F://tmp//data//raw data//train.npy")
label1=np.load("F://tmp//data//raw data//label_train.npy")
test=np.load("F://tmp//data//raw data//test.npy")
label1_test=np.load("F://tmp//data//raw data//label_test.npy")
for i in train:
    idx=random.sample(range(200),200) 
    i[idx] = 0

for i in test:
    idx=random.sample(range(200),200) 
    i[idx] = 0
label2 = []  
label2_test = []
for i in label1:
    if i[0] == 1:
        label2.append(1)
    if i[1] == 1:
        label2.append(2)
    if i[2] == 1:
        label2.append(3)
    if i[3] == 1:
        label2.append(4)

for i in label1_test:
    if i[0] == 1:
        label2_test.append(1)
    if i[1] == 1:
        label2_test.append(2)
    if i[2] == 1:
        label2_test.append(3)
    if i[3] == 1:
        label2_test.append(4)
clf = GaussianNB()
clf.fit(train, label2)
result1=clf.predict(test)
error_num1 = 0
for i in range(124):
    if result1[i] != label2_test[i]:
        error_num1 += 1

acc1 = (124-error_num1)/124.0

#KNN
from sklearn.neighbors import KNeighborsClassifier
train = np.load("F://tmp//data//raw data//train.npy")
label1=np.load("F://tmp//data//raw data//label_train.npy")
test=np.load("F://tmp//data//raw data//test.npy")
label1_test=np.load("F://tmp//data//raw data//label_test.npy")
for i in train:
    idx=random.sample(range(200),40) 
    i[idx] = 0

for i in test:
    idx=random.sample(range(200),40) 
    i[idx] = 0
label2 = []  
label2_test = []
for i in label1:
    if i[0] == 1:
        label2.append(1)
    if i[1] == 1:
        label2.append(2)
    if i[2] == 1:
        label2.append(3)
    if i[3] == 1:
        label2.append(4)

for i in label1_test:
    if i[0] == 1:
        label2_test.append(1)
    if i[1] == 1:
        label2_test.append(2)
    if i[2] == 1:
        label2_test.append(3)
    if i[3] == 1:
        label2_test.append(4)
#neigh = KNeighborsClassifier(n_neighbors=5)
#neigh.fit(train, label2)
result1=neigh.predict(test)
error_num1 = 0
for i in range(124):
    if result1[i] != label2_test[i]:
        error_num1 += 1

acc1 = (124-error_num1)/124.0





