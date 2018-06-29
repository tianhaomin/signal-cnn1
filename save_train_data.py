# -*- coding: utf-8 -*-
"""
Created on Sat May 19 14:19:49 2018

@author: Administrator
"""

import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn
import os
from pandas import DataFrame
import tensorflow as tf
import random
from sklearn import preprocessing 
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
############cdma_down#########2
data_cdma_down = read_data(850,870,880).reshape(850,401)
#############egsm_up#########3
#############egsm_down#########4
data_egsm_down = read_data(850,930,940).reshape(850,401)
############satellite##########
##############wlan##############5
data_wcdma_down = read_data(850,2135,2145).reshape(850,401)
##############4G################6
data_lte_down = read_data(850,1850,1860).reshape(850,401)
##############3G##################7

data_cdma_down = np.array( [[row[i] for i in range(0, 401) if i != 200] for row in data_cdma_down] )
data_egsm_down = np.array( [[row[i] for i in range(0, 401) if i != 200] for row in data_egsm_down] )
data_wcdma_down = np.array( [[row[i] for i in range(0, 401) if i != 200] for row in data_wcdma_down] )
data_lte_down = np.array( [[row[i] for i in range(0, 401) if i != 200] for row in data_lte_down] )
train_data = [data_cdma_down,data_egsm_down,data_wcdma_down,data_lte_down]
train = train_data[0]
for i in range(len(train_data)-1):
    train = np.vstack((train,train_data[i+1]))
#######test set#############
a = os.listdir("F:/project/Yin/spectrum-data")
def read_data(file_num,start_frq,end_frq):
    z1 = DataFrame({})
    for i in range(file_num):
        df = pandas.read_table("F:/project/Yin/spectrum-data//"+a[i+850],names=["fc","E"])
        df1 = df[df.fc.between(start_frq,end_frq)]
        z1 = pandas.concat([z1,df1['E']])
        z2 = z1.values
        z3 = np.array([z2])
    return z3

############cdma_down#########2
test_cdma_down = read_data(31,870,880).reshape(31,401)
#############egsm_up#########3
#############egsm_down#########4
test_egsm_down = read_data(31,930,940).reshape(31,401)
##########lte下行#########
test_lte_down = read_data(31,1850,1860).reshape(31,401)
#########wcdma下行###########
test_wcdma_down = read_data(31,2135,2145).reshape(31,401)
############satellite##########
##############wlan##############5
##############4G################6
##############3G##################7
#######
test_cdma_down = np.array( [[row[i] for i in range(0, 401) if i != 200] for row in test_cdma_down] )
test_egsm_down = np.array( [[row[i] for i in range(0, 401) if i != 200] for row in test_egsm_down] )
test_lte_down = np.array( [[row[i] for i in range(0, 401) if i != 200] for row in test_lte_down] )
test_wcdma_down = np.array( [[row[i] for i in range(0, 401) if i != 200] for row in test_wcdma_down] )
##############
test_data = [test_cdma_down,test_egsm_down,test_wcdma_down,test_lte_down]

test = test_data[0]
for i in range(len(test_data)-1):
    test = np.vstack((test,test_data[i+1]))
#######################################shuffle#######
train=train.astype(np.float32)
test = test.astype(np.float32)
enc = preprocessing.OneHotEncoder()  
enc.fit([[1],[2],[3],[4]])  

array1_1 = enc.transform([[1]]*850).toarray()  
array1_2 = enc.transform([[2]]*850).toarray()  
array1_3 = enc.transform([[3]]*850).toarray()  
array1_4 = enc.transform([[4]]*850).toarray()  
#array1_5 = enc.transform([[5]]*850).toarray()  
#array1_6 = enc.transform([[6]]*850).toarray()  
label1 = np.vstack((array1_1,array1_2,array1_3,array1_4))
array1_1_test = enc.transform([[1]]*31).toarray()
array1_2_test = enc.transform([[2]]*31).toarray()
array1_3_test = enc.transform([[3]]*31).toarray()
array1_4_test = enc.transform([[4]]*31).toarray()
#array1_5_test = enc.transform([[5]]*31).toarray()
#array1_6_test = enc.transform([[6]]*31).toarray()
label1_test = np.vstack((array1_1_test,array1_2_test,array1_3_test,array1_4_test))
# 顺序打乱
label1 = label1.astype(np.float32)
label1_test = label1_test.astype(np.float32)
idx=random.sample(range(3400),3400) 
idx1 = random.sample(range(124),124)
train = train[idx]
label1 = label1[idx]
test = test[idx1]
label1_test = label1_test[idx1]
np.save("F://tmp//data//raw data1//train.npy",train)
np.save("F://tmp//data//raw data1//test.npy",test)
np.save("F://tmp//data//raw data1//label1.npy",label1)
np.save("F://tmp//data//raw data1//label1_test.npy",label1_test)



