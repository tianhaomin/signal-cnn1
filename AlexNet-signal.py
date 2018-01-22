# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 19:40:23 2017

@author: Administrator
"""

#应用Alex网络进行识别建模

# read dataset
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
        z1 = pandas.concat([z1,df1[[1]]])
        z2 = z1['E']
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

data_cdma_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_cdma_down] )
data_egsm_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_egsm_down] )
data_wcdma_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_wcdma_down] )
data_lte_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_lte_down] )
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
        z1 = pandas.concat([z1,df1[[1]]])
        z2 = z1['E']
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
test_cdma_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_cdma_down] )
test_egsm_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_egsm_down] )
test_lte_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_lte_down] )
test_wcdma_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_wcdma_down] )
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



#Alex model 构建
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 200])
y_ = tf.placeholder(tf.float32, [None, 4])
x_image = tf.reshape(x,[-1,1,200,1])
with tf.name_scope('conv1') as scope:
    kernel = tf.Variable(tf.truncated_normal([1,3,1,64],dtype=tf.float32,stddev=1e-1),name='weights')
    conv = tf.nn.conv2d(x_image,kernel,[1,1,4,1],padding='SAME')
    biases = tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),trainable=True,name='biases')
    bias = tf.nn.bias_add(conv,biases)
    conv1 = tf.nn.relu(bias,name=scope)

lrn1 = tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn1')

pool1 = tf.nn.max_pool(lrn1,ksize=[1,1,3,1],strides=[1,1,2,1],padding='VALID',name='pool1')

with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(tf.truncated_normal([1,1,64,192],dtype=tf.float32,stddev=1e-1),name='weights')
    conv = tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='SAME')
    biases = tf.Variable(tf.constant(0.0,shape=[192],dtype=tf.float32),trainable=True,name='biases')
    bias = tf.nn.bias_add(conv,biases)
    conv2 = tf.nn.relu(bias,name=scope)

lrn2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn2')

pool2 = tf.nn.max_pool(lrn2,ksize=[1,1,3,1],strides=[1,1,2,1],padding='VALID',name='pool2')

with tf.name_scope('conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([1,3,192,384],dtype=tf.float32,stddev=1e-1),name='weights')
    conv = tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='SAME')
    biases = tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),trainable=True,name='biases')
    bias = tf.nn.bias_add(conv,biases)
    conv3 = tf.nn.relu(bias,name=scope)

with tf.name_scope('conv4') as scope:
    kernel = tf.Variable(tf.truncated_normal([1,3,384,256],dtype=tf.float32,stddev=1e-1),name='weights')
    conv = tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')
    biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
    bias = tf.nn.bias_add(conv,biases)
    conv4 = tf.nn.relu(bias,name=scope)

with tf.name_scope('conv5') as scope:
    kernel = tf.Variable(tf.truncated_normal([1,3,256,256],dtype=tf.float32,stddev=1e-1),name='weights')
    conv = tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding='SAME')
    biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
    bias = tf.nn.bias_add(conv,biases)
    conv5 = tf.nn.relu(bias,name=scope)

pool5 = tf.nn.max_pool(conv5,ksize=[1,1,3,1],strides=[1,1,2,1],padding='VALID',name='pool5')

pool5_flat = tf.reshape(pool5,[-1,1*5*256],name='pool5_flat')

with tf.name_scope('fc1') as scope:
    unites_w = tf.Variable(tf.truncated_normal([1*5*256,5000]),name='weights')
    unites_b = tf.Variable(tf.truncated_normal([5000]),name='biases')
    fc1 = tf.nn.relu(tf.matmul(pool5_flat,unites_w)+unites_b,name=scope)

with tf.name_scope('dropout') as scope:
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(fc1, keep_prob,name=scope)

with tf.name_scope('fc2') as scope:
    unites_w = tf.Variable(tf.truncated_normal([5000,4]),name='weights')
    unites_b = tf.Variable(tf.truncated_normal([4]),name='biases')
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,unites_w)+unites_b,name=scope)

#定义损失含税和优化的方法
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#正式用构建的模型进行分类
tf.global_variables_initializer().run()
for i in range(10000):
  a = random.randint(0,3200)
  #print(a)
  #train_step.run(feed_dict={x: train[i:i+1], y_: label1[i:i+1], keep_prob: 0.8})
  if i%10 == 0:#每训练100步进行一次输出，feed_dict就是对placeholder进行赋值
      train_accuracy = accuracy.eval(feed_dict={x: test, y_: label1_test, keep_prob: 1})
      print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: train[a:a+128], y_: label1[a:a+128], keep_prob: 0.8})
  #train_step.run(feed_dict={x: train[a:a+128], y_: label1[a:a+128], keep_prob: 0.8})






#在最终测试集上进行训练
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: test, y_: label1_test, keep_prob: 1.0}))



