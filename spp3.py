# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 20:15:56 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 15:33:56 2017

@author: Administrator
"""
#downlink 精简版本--无tensorbord版本自创的深层基本结构
#############good
import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn
import os
from pandas import DataFrame
import tensorflow as tf
import random
import math
from sklearn import preprocessing 


train = np.load("F://tmp//data//raw data1//train.npy")
label1=np.load("F://tmp//data//raw data1//label1.npy")
test=np.load("F://tmp//data//raw data1//test.npy")
label1_test=np.load("F://tmp//data//raw data1//label1_test.npy")

sess1  = tf.InteractiveSession()
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 1,-1,1],name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 4],name='y-input')
with tf.name_scope('input_reshape'):
    x_image = tf.reshape(x,[-1,1,-1,1])
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
with tf.name_scope('conv_1'):
    with tf.name_scope('weights'):
        W_conv1 = weight_variable([1, 1, 1, 32])
    with tf.name_scope('biases'):
        b_conv1 = bias_variable([32])
    with tf.name_scope('conv'):
        conv1 = conv2d(x_image, W_conv1)
    h_conv1 = tf.nn.relu(conv1+b_conv1,name='activation')
with tf.name_scope('conv_2'):
    with tf.name_scope('weights'):
        W_conv2 = weight_variable([1, 3, 32, 48])
    with tf.name_scope('biases'):
        b_conv2 = bias_variable([48])
    with tf.name_scope('conv'):
        conv2 = conv2d(h_conv1, W_conv2)
    h_conv2 = tf.nn.relu(conv2+b_conv2,name='activation')
with tf.name_scope('maxpool_3'):
    h_pool3 = max_pool_2x2(h_conv2)
with tf.name_scope('conv_4'):
    with tf.name_scope('weights'):
        W_conv4 = weight_variable([1, 1, 48, 64])
    with tf.name_scope('biases'):
        b_conv4 = bias_variable([64])
    with tf.name_scope('conv'):
        conv4 = conv2d(h_pool3, W_conv4)
    h_conv4 = tf.nn.relu(conv4+b_conv4,name='activation')
with tf.name_scope('conv_5'):
    with tf.name_scope('weights'):
        W_conv5 = weight_variable([1, 3, 64, 80])
    with tf.name_scope('biases'):
        b_conv5 = bias_variable([80])
    with tf.name_scope('conv'):
        conv5 = conv2d(h_conv4, W_conv5)
    h_conv5 = tf.nn.relu(conv5+b_conv5,name='activation')
with tf.name_scope('maxpool_5'):
    h_pool5 = max_pool_2x2(h_conv5)
with tf.name_scope('conv_6'):
    with tf.name_scope('weights'):
        W_conv6 = weight_variable([1, 1, 80, 128])
    with tf.name_scope('biases'):
        b_conv6 = bias_variable([128])
    with tf.name_scope('conv'):
        conv6 = conv2d(h_pool5, W_conv6)
        tf.summary.histogram('conv',conv6)
    h_conv6 = tf.nn.relu(conv6+b_conv6,name='activation')
with tf.name_scope('conv_7'):
    with tf.name_scope('weights'):
        W_conv7 = weight_variable([1, 3, 128, 156])
    with tf.name_scope('biases'):
        b_conv7 = bias_variable([156])
    with tf.name_scope('conv'):
        conv7 = conv2d(h_conv6, W_conv7)
    h_conv7 = tf.nn.relu(conv7+b_conv7,name='activation')
with tf.name_scope('maxpool_8'):
    h_pool8 = max_pool_2x2(h_conv7)
with tf.name_scope('conv_9'):
    with tf.name_scope('weights'):
        W_conv9 = weight_variable([1, 3, 156, 200])
    with tf.name_scope('biases'):
        b_conv9 = bias_variable([200])
    with tf.name_scope('conv'):
        conv9 = conv2d(h_pool8, W_conv9)
    h_conv9 = tf.nn.relu(conv9+b_conv9,name='activation')
with tf.name_scope('conv_10'):
    with tf.name_scope('weights'):
        W_conv10 = weight_variable([1, 3, 200, 256])
    with tf.name_scope('biases'):
        b_conv10 = bias_variable([256])
    with tf.name_scope('conv'):
        conv10 = conv2d(h_conv9, W_conv10)
        tf.summary.histogram('conv',conv10)
    h_conv10 = tf.nn.relu(conv10+b_conv10,name='activation')
with tf.name_scope('conv_11'):
    with tf.name_scope('weights'):
        W_conv11 = weight_variable([1, 3, 256, 300])
    with tf.name_scope('biases'):
        b_conv11 = bias_variable([300])
    with tf.name_scope('conv'):
        conv11 = conv2d(h_conv10, W_conv11)
    h_conv11 = tf.nn.relu(conv11+b_conv11,name='activation')
with tf.name_scope('SPP_Pooling'):
    bins=[8,4,1]
    map_size = h_conv11.get_shape().as_list()[2]
    strides1 = []
    filter1 = []
    pool_out = []
    for i in range(len(bins)):
        x1 = int(math.floor(float(map_size)/bins[i]))
        strides1.append(x1)
        x1 = int(math.ceil(float(map_size)/bins[i]))
        filter1.append(x1)
    for i in range(len(bins)):
        pool_out.append(tf.nn.max_pool(h_conv11,ksize=[1,1,filter1[i],1],strides=[1,1,strides1[i],1],padding='VALID'))
    output = tf.concat([pool_out[0], pool_out[1], pool_out[2]],2)
h_pool12_flat = tf.reshape(output, [-1, 1*13*300])
with tf.name_scope('fc_1'):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
        W_fc1 = weight_variable([1* 13 * 300, 5000])
    with tf.name_scope('biases'):
        b_fc1 = bias_variable([5000])
    with tf.name_scope('Wx_plus_b'):
        preactivate1 = tf.matmul(h_pool12_flat, W_fc1) + b_fc1
    activations1 = tf.nn.relu(preactivate1, name='activation')
#keep_prob = tf.placeholder(tf.float32)
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32,name='dropout')
    h_fc1_drop = tf.nn.dropout(activations1, keep_prob)
with tf.name_scope('fc_2'):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
        W_fc2 = weight_variable([5000,4])
    with tf.name_scope('biases'):
        b_fc2 = bias_variable([4])
    with tf.name_scope('Wx_plus_b'):
        preactivate2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        tf.summary.histogram('pre_activations', preactivate2)
    activations2 = tf.nn.softmax(preactivate2, name='activation')
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(activations2,1e-10,1.0)), reduction_indices=[1]))
#tf.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(activations2, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
for i in range(3000):
    for j in range(67):
        if j % 10 == 0:  # Record summaries and test-set accuracy
            acc = sess1.run([accuracy], feed_dict={x:test,y_:label1_test,keep_prob:1.0})
            print('Accuracy at step %s of epoch %s: %s' % (j,i, acc))
        else:  # Record a summary
    #         t1=time.time()
            _ = sess1.run([train_step], feed_dict={x:train[50*j:50*(j+1)],y_:label1[50*j:50*(j+1)],keep_prob:0.8})
            acc_train = sess1.run([accuracy],feed_dict={x:train[0:200],y_:label1[0:200],keep_prob:0.8})
            print(acc_train)


sess1.close()




