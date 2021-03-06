# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 22:47:27 2017

@author: Administrator
"""
################nmodel1###############
import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn
import os
from pandas import DataFrame
import tensorflow as tf
import random
from sklearn import preprocessing 

log_dir1 = 'D:/temp/tensorflow/project/logs/projectcnn_with_summaries1_1'
enc = preprocessing.OneHotEncoder()  
enc.fit([[1]])  

array1_1 = enc.transform([[1]]*1700).toarray()  
array1_2 = enc.transform([[0]]*1700).toarray()  
label1 = np.vstack((array1_1,array1_2))
array1_1_test = enc.transform([[1]]*62).toarray()
array1_2_test = enc.transform([[0]]*62).toarray()
label1_test = np.vstack((array1_1_test,array1_2_test))
# 顺序打乱
label1 = label1[idx]
label1 = label1.astype(np.float32)
label1_test = label1_test[idx1]
label1_test = label1_test.astype(np.float32)




sess1  = tf.InteractiveSession()
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 200],name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 1],name='y-input')
with tf.name_scope('input_reshape'):
    x_image = tf.reshape(x,[-1,2,100,1])
    tf.summary.image('input', x_image, 2)
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
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
with tf.name_scope('conv_1'):
    with tf.name_scope('weights'):
        W_conv1 = weight_variable([1, 1, 1, 32])
        variable_summaries(W_conv1)
    with tf.name_scope('biases'):
        b_conv1 = bias_variable([32])
        variable_summaries(b_conv1)
    with tf.name_scope('conv'):
        conv1 = conv2d(x_image, W_conv1)
        tf.summary.histogram('conv',conv1)
    h_conv1 = tf.nn.relu(conv1+b_conv1,name='activation')
    tf.summary.histogram('h_conv1', h_conv1)
with tf.name_scope('conv_2'):
    with tf.name_scope('weights'):
        W_conv2 = weight_variable([1, 3, 32, 64])
        variable_summaries(W_conv2)
    with tf.name_scope('biases'):
        b_conv2 = bias_variable([64])
        variable_summaries(b_conv2)
    with tf.name_scope('conv'):
        conv2 = conv2d(h_conv1, W_conv2)
        tf.summary.histogram('conv',conv2)
    h_conv2 = tf.nn.relu(conv2+b_conv2,name='activation')
    tf.summary.histogram('h_conv2', h_conv2)
with tf.name_scope('conv_3'):
    with tf.name_scope('weights'):
        W_conv3 = weight_variable([1, 1, 64, 128])
        variable_summaries(W_conv3)
    with tf.name_scope('biases'):
        b_conv3 = bias_variable([128])
        variable_summaries(b_conv3)
    with tf.name_scope('conv'):
        conv3 = conv2d(h_conv2, W_conv3)
        tf.summary.histogram('conv',conv3)
    h_conv3 = tf.nn.relu(conv3+b_conv3,name='activation')
    tf.summary.histogram('h_conv3', h_conv3)
with tf.name_scope('maxpool_3'):
    h_pool3 = max_pool_2x2(h_conv3)
    tf.summary.histogram('h_pool3', h_pool3)
with tf.name_scope('conv_4'):
    with tf.name_scope('weights'):
        W_conv4 = weight_variable([1, 3, 128, 128])
        variable_summaries(W_conv4)
    with tf.name_scope('biases'):
        b_conv4 = bias_variable([128])
        variable_summaries(b_conv4)
    with tf.name_scope('conv'):
        conv4 = conv2d(h_pool3, W_conv4)
        tf.summary.histogram('conv',conv4)
    h_conv4 = tf.nn.relu(conv4+b_conv4,name='activation')
    tf.summary.histogram('h_conv4', h_conv4)
with tf.name_scope('conv_5'):
    with tf.name_scope('weights'):
        W_conv5 = weight_variable([1, 3, 128, 128])
        variable_summaries(W_conv5)
    with tf.name_scope('biases'):
        b_conv5 = bias_variable([128])
        variable_summaries(b_conv5)
    with tf.name_scope('conv'):
        conv5 = conv2d(h_conv4, W_conv5)
        tf.summary.histogram('conv',conv5)
    h_conv5 = tf.nn.relu(conv5+b_conv5,name='activation')
    tf.summary.histogram('h_conv5', h_conv5)
with tf.name_scope('maxpool_5'):
    h_pool5 = max_pool_2x2(h_conv5)
    tf.summary.histogram('h_pool5', h_pool5)
print(h_pool5.shape)
h_pool6_flat = tf.reshape(h_pool5, [-1, 1*25*128])
with tf.name_scope('fc_1'):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
        W_fc1 = weight_variable([1* 25 * 128, 2000])
        variable_summaries(W_fc1)
    with tf.name_scope('biases'):
        b_fc1 = bias_variable([2000])
        variable_summaries(b_fc1)
    with tf.name_scope('Wx_plus_b'):
        preactivate1 = tf.matmul(h_pool6_flat, W_fc1) + b_fc1
        tf.summary.histogram('pre_activations', preactivate1)
    activations1 = tf.nn.relu(preactivate1, name='activation')
    tf.summary.histogram('activations', activations1)
#keep_prob = tf.placeholder(tf.float32)
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    h_fc1_drop = tf.nn.dropout(activations1, keep_prob)
with tf.name_scope('fc_2'):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
        W_fc2 = weight_variable([2000,2])
        variable_summaries(W_fc2)
    with tf.name_scope('biases'):
        b_fc2 = bias_variable([2])
        variable_summaries(b_fc2)
    with tf.name_scope('Wx_plus_b'):
        preactivate2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        tf.summary.histogram('pre_activations', preactivate2)
    activations2 = tf.nn.softmax(preactivate2, name='activation')
    tf.summary.histogram('activations', activations2)
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(activations2), reduction_indices=[1]))
tf.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(activations2, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)



merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir1 + '/train', sess1.graph)
test_writer = tf.summary.FileWriter(log_dir1 + '/test')
tf.global_variables_initializer().run()



saver = tf.train.Saver()  
for i in range(50000):
    a = random.randint(0,3200)
    if i % 10 == 0:  # Record summaries and test-set accuracy
        summary, acc = sess1.run([merged, accuracy], feed_dict={x:test,y_:label1_test,keep_prob:float(1.0)})
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
    else:
        # Record train set summaries, and train
        if i % 100 == 99:  # Record execution stats
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess1.run([merged, train_step],
                            feed_dict={x:train[a:a+128],y_:label1[a:a+128],keep_prob:0.8},
                            options=run_options,
                            run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            saver.save(sess1, log_dir1+"/model.ckpt", i)
            print('Adding run metadata for', i)
        else:  # Record a summary
            summary, _ = sess1.run([merged, train_step], feed_dict={x:train[a:a+128],y_:label1[a:a+128],keep_prob:0.8})
            train_writer.add_summary(summary, i)
            #print(2)
train_writer.close()
test_writer.close()


sess1.close()