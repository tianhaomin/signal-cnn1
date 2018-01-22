# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:40:13 2017

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
# 堆取数据
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
##########进行频谱切片##############2M为一个切片
#所有预准备的频谱样本用2M的宽度进行切片
data_cdma_up = read_data(850,825,835)
data_cdma_down = read_data(850,870,880)
data_egsm_up = read_data(850,885,909)
data_egsm_down = read_data(850,903,954)
data_setallit = read_data(850,1980,2010)
data_wlan = read_data(850,2400,2483)
data_4G = read_data(850,1880,1890)
data_3G = read_data(850,2010,2025)
##################
###########cdma_up########
data_cdma_up1 = read_data(850,825,827).reshape(850,81)
data_cdma_up2 = read_data(850,827,829).reshape(850,81)
data_cdma_up3 = read_data(850,829,831).reshape(850,81)
data_cdma_up4 = read_data(850,831,833).reshape(850,81)
data_cdma_up5 = read_data(850,833,835).reshape(850,81)
############cdma_down#########
data_cdma_down1 = read_data(850,870,872).reshape(850,81)
data_cdma_down2 = read_data(850,872,874).reshape(850,81)
data_cdma_down3 = read_data(850,874,876).reshape(850,81)
data_cdma_down4 = read_data(850,876,878).reshape(850,81)
data_cdma_down5 = read_data(850,878,880).reshape(850,81)
#############egsm_up#########
data_egsm_up1 = read_data(850,885,887).reshape(850,81)
data_egsm_up2 = read_data(850,887,889).reshape(850,81)
data_egsm_up3 = read_data(850,889,891).reshape(850,81)
data_egsm_up4 = read_data(850,891,893).reshape(850,81)
data_egsm_up5 = read_data(850,893,895).reshape(850,81)
#############egsm_down#########
data_egsm_down1 = read_data(850,903,905).reshape(850,81)
data_egsm_down2 = read_data(850,905,907).reshape(850,81)
data_egsm_down3 = read_data(850,907,909).reshape(850,81)
data_egsm_down4 = read_data(850,909,911).reshape(850,81)
data_egsm_down5 = read_data(850,911,913).reshape(850,81)
############satellite##########
##############wlan##############
data_wlan1 = read_data(850,2400,2402).reshape(850,81)
data_wlan2 = read_data(850,2403,2405).reshape(850,81)
data_wlan3 = read_data(850,2405,2407).reshape(850,81)
data_wlan4 = read_data(850,2479,2481).reshape(850,81)
data_wlan5 = read_data(850,2481,2483).reshape(850,81)
##############4G################
data_lte1 = read_data(850,1880,1882).reshape(850,81)
data_lte2 = read_data(850,1882,1884).reshape(850,81)
data_lte3 = read_data(850,1884,1886).reshape(850,81)
data_lte4 = read_data(850,1886,1888).reshape(850,81)
data_lte5 = read_data(850,1888,1890).reshape(850,81)
##############3G##################
data_scdma1 = read_data(850,2010.5,2012.5).reshape(850,81)
data_scdma2 = read_data(850,2013.7,2015.7).reshape(850,81)
data_scdma3 = read_data(850,2017.1,2019.1).reshape(850,81)
data_scdma4 = read_data(850,2020.3,2022.3).reshape(850,81)
data_scdma5 = read_data(850,2023.5,2025.5).reshape(850,81)
#########training set############
train_data = [data_cdma_up1,data_cdma_up2,data_cdma_up3,data_cdma_down1,data_cdma_down2,
                        data_cdma_down3,data_egsm_up1,data_egsm_up2,data_egsm_up3,data_egsm_down1,
                        data_egsm_down2,data_egsm_down3,data_wlan1,data_wlan2,data_wlan3,
                        data_lte1,data_lte2,data_lte3,data_scdma1,data_scdma2,data_scdma3]
train = train_data[0]
for i in range(len(train_data)-1):
    train = np.vstack((train,train_data[i+1]))
#########valid_set##############
valid_data = [data_cdma_up4,data_cdma_down4,data_egsm_up4,data_egsm_down4,data_wlan4,data_lte4,data_scdma4]
valid = valid_data[0]
for i in range(len(valid_data)-1):
    valid = np.vstack((valid,valid_data[i+1]))
#########test################
test_data = [data_cdma_up5,data_cdma_down5,data_egsm_up5,data_egsm_down5,data_wlan5,data_lte5,data_scdma5]
test = test_data[0]
for i in range(len(test_data)-1):
    test = np.vstack((test,test_data[i+1]))
############label_train##############
from sklearn import preprocessing  

enc = preprocessing.OneHotEncoder()  
enc.fit([[1],[2],[3],[4],[5],[6],[7]])  

array1 = enc.transform([[1]]*2550).toarray()  
array2 = enc.transform([[2]]*2550).toarray()  
array3 = enc.transform([[3]]*2550).toarray()  
array4 = enc.transform([[4]]*2550).toarray()  
array5 = enc.transform([[5]]*2550).toarray()  
array6 = enc.transform([[6]]*2550).toarray() 
array7 = enc.transform([[7]]*2550).toarray()   
label_train = np.vstack((array1,array2,array3,array4,array5,array6,array7))
##################label_valid#########
array1 = enc.transform([[1]]*850).toarray()  
array2 = enc.transform([[2]]*850).toarray()  
array3 = enc.transform([[3]]*850).toarray()  
array4 = enc.transform([[4]]*850).toarray()  
array5 = enc.transform([[5]]*850).toarray()  
array6 = enc.transform([[6]]*850).toarray() 
array7 = enc.transform([[7]]*850).toarray()   
label_valid = np.vstack((array1,array2,array3,array4,array5,array6,array7))
###################label_test###########
array1 = enc.transform([[1]]*850).toarray()  
array2 = enc.transform([[2]]*850).toarray()  
array3 = enc.transform([[3]]*850).toarray()  
array4 = enc.transform([[4]]*850).toarray()  
array5 = enc.transform([[5]]*850).toarray()  
array6 = enc.transform([[6]]*850).toarray() 
array7 = enc.transform([[7]]*850).toarray()   
label_test = np.vstack((array1,array2,array3,array4,array5,array6,array7))
################
idx=random.sample(range(17850),17850) 
train = train[idx]
label_train = label_train[idx]




###############model set_up ##################
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import random
#导入数据创建爱你交互模式

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
x = tf.placeholder(tf.float32, [None, 81])
y_ = tf.placeholder(tf.float32, [None, 7])
x_image = tf.reshape(x,[-1,1,81,1])
#定义第一个卷积层  
W_conv1 = weight_variable([1, 1, 1, 64])
b_conv1 = bias_variable([64])
#权重和偏置设置好就开始做卷积处理conv2d(x,w)然后进行RELU非线性激活处理
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
                     
W_conv2 = weight_variable([1, 3, 64, 64])
b_conv2 = bias_variable([64])
#权重和偏置设置好就开始做卷积处理conv2d(x,w)然后进行RELU非线性激活处理
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
#最大池化处理
h_pool2 = max_pool_2x2(h_conv2)
#同上第二个卷积层的定义，前一层有32个卷积核这一层的通道数就有32
W_conv3 = weight_variable([1, 1, 64, 64])
b_conv3 = bias_variable([64])
#权重和偏置设置好就开始做卷积处理conv2d(x,w)然后进行RELU非线性激活处理
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

W_conv4 = weight_variable([1, 3, 64, 128])
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

W_conv5 = weight_variable([1, 1, 128, 128])
b_conv5 = bias_variable([128])
#权重和偏置设置好就开始做卷积处理conv2d(x,w)然后进行RELU非线性激活处理
h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)

W_conv6 = weight_variable([1,3,128,256])
b_conv6 = bias_variable([256])
h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)
h_pool6 = max_pool_2x2(h_conv6)
#因为前面经历了两次的2*2 的最大池化反应，所以边长只有1/4图片尺寸变成了7*7而
#第二个卷积核的数量有64个所以输出的tensor的尺寸是7*7*64
#我们需要将输出的tensor进行变形整合成一维向量然后连接一个全连接
#隐藏节点是1024并且使用RELU进行激活

W_fc1 = weight_variable([1* 3 * 256, 1024])
b_fc1 = bias_variable([1024])
h_pool6_flat = tf.reshape(h_pool6, [-1, 1*3*256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool6_flat, W_fc1) + b_fc1)
#用dropout层减少过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#用softmax实现多分类
W_fc2 = weight_variable([1024, 7])
b_fc2 = bias_variable([7])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#定义损失含税和优化的方法
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#正式用构建的模型进行分类
tf.global_variables_initializer().run()
for i in range(10000):
  a = random.randint(0,17799)
  if i%100 == 0:#每训练100步进行一次输出，feed_dict就是对placeholder进行赋值
    train_accuracy = accuracy.eval(feed_dict={
        x:train[a:a+50], y_: label_train[a:a+50], keep_prob: 1})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: train[a:a+50], y_: label_train[a:a+50], keep_prob: 1})






#在最终测试集上进行训练
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))




