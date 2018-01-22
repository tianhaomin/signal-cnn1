# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:43:13 2017

@author: Administrator
"""
####good
import os
from pandas import DataFrame
import tensorflow as tf
import random
from sklearn import preprocessing 
import numpy as np
import pandas
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
data_cdma_up1 = read_data(850,825,830).reshape(850,201)
data_cdma_up2 = read_data(850,830,835).reshape(850,201)
############cdma_down#########2
data_cdma_down1 = read_data(850,870,875).reshape(850,201)
data_cdma_down2 = read_data(850,875,880).reshape(850,201)
data_cdma_down1 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_cdma_down1] )
data_cdma_down2 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_cdma_down2] )
data_cdma_up1 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_cdma_up1] )
data_cdma_up2 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_cdma_up2] )
train_data = [data_cdma_up1,data_cdma_up2,data_cdma_down1,data_cdma_down2]
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

test_cdma_up1 = read_data(31,825,830).reshape(31,201)
test_cdma_up2 = read_data(31,830,835).reshape(31,201)
############cdma_down#########2
test_cdma_down1 = read_data(31,870,875).reshape(31,201)
test_cdma_down2 = read_data(31,875,880).reshape(31,201)
test_cdma_down1 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_cdma_down1] )
test_cdma_down2 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_cdma_down2] )
test_cdma_up1 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_cdma_up1] )
test_cdma_up2 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_cdma_up2] )
test_data = [test_cdma_up1,test_cdma_up2,test_cdma_down1,test_cdma_down2]
test = test_data[0]
for i in range(len(test_data)-1):
    test = np.vstack((test,test_data[i+1]))
#######################################shuffle#######
idx=random.sample(range(1700),1700) 
idx1 = random.sample(range(62),62)
train = train[idx]
train=train.astype(np.float32)
test = test[idx1]
test = test.astype(np.float32)
enc = preprocessing.OneHotEncoder()  
enc.fit([[1],[2]])  

array1_1 = enc.transform([[1]]*1700).toarray()  
array1_2 = enc.transform([[2]]*1700).toarray()  
label1 = np.vstack((array1_1,array1_2))
array1_1_test = enc.transform([[1]]*62).toarray()
array1_2_test = enc.transform([[2]]*62).toarray()
label1_test = np.vstack((array1_1_test,array1_2_test))
# 顺序打乱
label1 = label1[idx]
label1 = label1.astype(np.float32)
label1_test = label1_test[idx1]
label1_test = label1_test.astype(np.float32)

sess = tf.InteractiveSession()

#定义权重初始化函数
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
#定义偏置的初始化函数
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#定义二维的卷积层 ，W是卷积层的参数[a,b,c,d](a,b)是卷积核的大小，c代表图片的通道数，
#d代表卷积核的数量也就是这层卷积会提取多少个特征
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#最大池化层
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 3, 3, 1], padding='SAME')  
#规定输入输出的格式，因为原先是得到784维的向量而卷机网络是处理
#二维的图像所以将向量转为28*28的格式                       
x = tf.placeholder(tf.float32, [None, 200])
y_ = tf.placeholder(tf.float32, [None, 2])
x_image = tf.reshape(x,[-1,1,200,1])
#定义第一个卷积层  
W_conv1 = weight_variable([1, 1, 1, 32])
b_conv1 = bias_variable([32])
#权重和偏置设置好就开始做卷积处理conv2d(x,w)然后进行RELU非线性激活处理
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
                     
W_conv2 = weight_variable([1, 3, 32, 48])
b_conv2 = bias_variable([48])
#权重和偏置设置好就开始做卷积处理conv2d(x,w)然后进行RELU非线性激活处理
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
#最大池化处理
h_pool2 = max_pool_2x2(h_conv2)
#同上第二个卷积层的定义，前一层有32个卷积核这一层的通道数就有32
W_conv3 = weight_variable([1, 1, 48, 64])
b_conv3 = bias_variable([64])
#权重和偏置设置好就开始做卷积处理conv2d(x,w)然后进行RELU非线性激活处理
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

W_conv4 = weight_variable([1, 3, 64, 80])
b_conv4 = bias_variable([80])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

W_conv5 = weight_variable([1, 1, 80, 128])
b_conv5 = bias_variable([128])
#权重和偏置设置好就开始做卷积处理conv2d(x,w)然后进行RELU非线性激活处理
h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)

W_conv6 = weight_variable([1,3,128,156])
b_conv6 = bias_variable([156])
h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)
h_pool6 = max_pool_2x2(h_conv6)

print(h_pool6.shape)
#因为前面经历了两次的2*2 的最大池化反应，所以边长只有1/4图片尺寸变成了7*7而
#第二个卷积核的数量有64个所以输出的tensor的尺寸是7*7*64
#我们需要将输出的tensor进行变形整合成一维向量然后连接一个全连接
#隐藏节点是1024并且使用RELU进行激活

W_fc1 = weight_variable([1* 8 * 156, 3000])
b_fc1 = bias_variable([3000])
h_pool6_flat = tf.reshape(h_pool6, [-1, 1*156*8])
h_fc1 = tf.nn.relu(tf.matmul(h_pool6_flat, W_fc1) + b_fc1)
#用dropout层减少过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#用softmax实现多分类
W_fc2 = weight_variable([3000, 2])
b_fc2 = bias_variable([2])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#定义损失含税和优化的方法
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
#train_step = tf.train.AdamOptimizer(1e-7).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#正式用构建的模型进行分类
tf.global_variables_initializer().run()
for i in range(10000):
  a = random.randint(0,3200)
  #print(a)
  #train_step.run(feed_dict={x: train[i:i+1], y_: label1[i:i+1], keep_prob: 0.8})
  if i%10 == 0:#每训练100步进行一次输出，feed_dict就是对placeholder进行赋值
    train_accuracy = accuracy.eval(feed_dict={
        x: test, y_: label1_test, keep_prob: 1})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: train[a:a+128], y_: label1[a:a+128], keep_prob: 0.8})
  #train_step.run(feed_dict={x: train[a:a+128], y_: label1[a:a+128], keep_prob: 0.8})






#在最终测试集上进行训练
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: test, y_: label1_test, keep_prob: 1.0}))
