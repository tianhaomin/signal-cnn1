# -*- coding: utf-8 -*-
"""
Created on Fri May 18 11:31:42 2018

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
import math
train1 = np.load("F://tmp//data//raw data//train.npy")
label1=np.load("F://tmp//data//raw data//label_train.npy")
test=np.load("F://tmp//data//raw data//test.npy")
label1_test=np.load("F://tmp//data//raw data//label_test.npy")
train1 = train1.reshape(3400,1,200,1)
test = test.reshape(124,1,200,1)
sess1  = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
#定义偏置的初始化函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#最大池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')  

def interference(train,bins=[8,3,1],keep_prob=0.5):
	W_conv1 = weight_variable([1, 1, 1, 32])
	b_conv1 = bias_variable([32])
	conv1 = conv2d(train, W_conv1)
	h_conv1 = tf.nn.relu(conv1+b_conv1,name='activation')
	W_conv2 = weight_variable([1, 3, 32, 48])
	b_conv2 = bias_variable([48])
	conv2 = conv2d(h_conv1, W_conv2)
	h_conv2 = tf.nn.relu(conv2+b_conv2,name='activation')
	h_pool3 = max_pool_2x2(h_conv2)
	W_conv4 = weight_variable([1, 1, 48, 64])
	b_conv4 = bias_variable([64])
	conv4 = conv2d(h_pool3, W_conv4)
	h_conv4 = tf.nn.relu(conv4+b_conv4,name='activation')
	W_conv5 = weight_variable([1, 3, 64, 80])
	b_conv5 = bias_variable([80])
	conv5 = conv2d(h_conv4, W_conv5)
	h_conv5 = tf.nn.relu(conv5+b_conv5,name='activation')
	h_pool5 = max_pool_2x2(h_conv5)
	W_conv6 = weight_variable([1, 1, 80, 128])
	b_conv6 = bias_variable([128])
	conv6 = conv2d(h_pool5, W_conv6)
	h_conv6 = tf.nn.relu(conv6+b_conv6,name='activation')
	W_conv7 = weight_variable([1, 3, 128, 156])
	b_conv7 = bias_variable([156])
	conv7 = conv2d(h_conv6, W_conv7)
	h_conv7 = tf.nn.relu(conv7+b_conv7,name='activation')
	h_pool8 = max_pool_2x2(h_conv7)
	W_conv9 = weight_variable([1, 3, 156, 200])
	b_conv9 = bias_variable([200])
	conv9 = conv2d(h_pool8, W_conv9)
	h_conv9 = tf.nn.relu(conv9+b_conv9,name='activation')
	W_conv10 = weight_variable([1, 3, 200, 256])
	b_conv10 = bias_variable([256])
	conv10 = conv2d(h_conv9, W_conv10)
	h_conv10 = tf.nn.relu(conv10+b_conv10,name='activation')
	W_conv11 = weight_variable([1, 3, 256, 300])
	b_conv11 = bias_variable([300])
	conv11 = conv2d(h_conv10, W_conv11)
	h_conv11 = tf.nn.relu(conv11+b_conv11,name='activation')
	map_size = h_conv11.get_shape().as_list()[2]
	strides1 = []
	filter1 = []
	pool_out = []
	for i in range(len(bins)):
		x = int(math.floor(float(map_size)/bins[i]))
		strides1.append(x)
		x = int(math.ceil(float(map_size)/bins[i]))
		filter1.append(x)
	for i in range(len(bins)):
		pool_out.append(tf.nn.max_pool(h_conv11,ksize=[1,1,filter1[i],1],strides=[1,1,strides1[i],1],padding='VALID'))
	output = tf.concat([pool_out[0], pool_out[1], pool_out[2]],2)

	h_pool12_flat = tf.reshape(output, [-1, 1*12*300])
	W_fc1 = weight_variable([3600, 5000])
	b_fc1 = bias_variable([5000])
	preactivate1 = tf.matmul(h_pool12_flat, W_fc1) + b_fc1
	activations1 = tf.nn.relu(preactivate1, name='activation')
	h_fc1_drop = tf.nn.dropout(activations1, keep_prob)
	W_fc2 = weight_variable([5000,4])
	b_fc2 = bias_variable([4])
	preactivate2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	activations2 = tf.nn.softmax(preactivate2, name='activation')
	return activations2

def loss(activations2,label_1):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(label_1 * tf.log(tf.clip_by_value(activations2,1e-10,1.0)), reduction_indices=[1]))
	return cross_entropy
	
def train(loss):
	train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)

def acc(activations2,label_1):
	correct_prediction = tf.equal(tf.argmax(activations2, 1), tf.argmax(label_1, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#	accuracy = sess1.run(accuracy)
	return accuracy

#y_ = tf.placeholder(tf.float32, [None, 4],name='y-input')
#activation = tf.placeholder
#with tf.name_scope('cross_entropy'):
#    cross_entropy = tf.reduce_mean(-tf.reduce_sum(label1 * tf.log(tf.clip_by_value(activations2,1e-10,1.0)), reduction_indices=[1]))
#with tf.name_scope('train'):
#    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
#with tf.name_scope('accuracy'):
#    with tf.name_scope('correct_prediction'):
#        correct_prediction = tf.equal(tf.argmax(activations2, 1), tf.argmax(label1, 1))
#    with tf.name_scope('accuracy'):
#        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()

for i in range(3000):
    for j in range(67):
        if j % 10 == 0:  
            activation1 = interference(test)
            acc_test = acc(activation1,label1_test)
            print(acc_test.eval())
            print('Accuracy at step %s of epoch %s: %s' % (j,i, acc_test))
        else:  
            activation = interference(train1[50*j:50*(j+1)])
            entroloss = loss(activation,label1[50*j:50*(j+1)])
            train(entroloss)
            acc_train = acc(activation,label1[50*j:50*(j+1)])
            print(acc_train.eval)
            result_train.append(acc_train.eval())
#            _ = sess1.run([train_step], feed_dict={x:train[50*j:50*(j+1)],y_:label1[50*j:50*(j+1)],keep_prob:0.8})
#            acc_train = sess1.run([accuracy],feed_dict={x:train[0:200],y_:label1[0:200],keep_prob:0.8})
#            result_train.append(acc_train)
#            print(acc_train)


sess1.close()