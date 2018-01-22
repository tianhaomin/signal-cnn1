# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:56:24 2017

@author: Administrator
"""
##################多个CNN协同处理信号分类##################
################################################step1:统一数据处理################################
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
data_cdma_up1 = read_data(850,825,830).reshape(850,201)
data_cdma_up2 = read_data(850,830,835).reshape(850,201)
############cdma_down#########2
data_cdma_down1 = read_data(850,870,875).reshape(850,201)
data_cdma_down2 = read_data(850,875,880).reshape(850,201)
#############egsm_up#########3
data_egsm_up1 = read_data(850,885,890).reshape(850,201)
data_egsm_up2 = read_data(850,890,895).reshape(850,201)
#############egsm_down#########4
data_egsm_down1 = read_data(850,903,908).reshape(850,201)
data_egsm_down2 = read_data(850,908,913).reshape(850,201)
############satellite##########
##############wlan##############5
data_wlan1 = read_data(850,2400,2405).reshape(850,201)
data_wlan2 = read_data(850,2405,2410).reshape(850,201)
##############4G################6
data_lte1 = read_data(850,1880,1885).reshape(850,201)
data_lte2 = read_data(850,1885,1890).reshape(850,201)
##############3G##################7
data_scdma1 = read_data(850,2010.5,2015.5).reshape(850,201)
data_scdma2 = read_data(850,2015.5,2020.5).reshape(850,201)
data_cdma_down1 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_cdma_down1] )
data_cdma_down2 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_cdma_down2] )
data_cdma_up1 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_cdma_up1] )
data_cdma_up2 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_cdma_up2] )
data_lte1 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_lte1] )
data_lte2 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_lte2] )
data_scdma1 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_scdma1] )
data_scdma2 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_scdma2] )
data_wlan1 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_wlan1] )
data_wlan2 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_wlan2] )
data_egsm_up1 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_egsm_up1] )
data_egsm_up2 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_egsm_up2] )
data_egsm_down1 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_egsm_down1] )
data_egsm_down2 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_egsm_down2] )
train_data = [data_cdma_up1,data_cdma_up2,data_cdma_down1,data_cdma_down2,
                    data_egsm_up1,data_egsm_up2,data_egsm_down1,
                        data_egsm_down2,data_wlan1,data_wlan2,
                        data_lte1,data_lte2,data_scdma1,data_scdma2]
train_data = [data_egsm_up1,data_egsm_up2,data_egsm_down1,
                        data_egsm_down2]
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
#############egsm_up#########3
test_egsm_up1 = read_data(31,885,890).reshape(31,201)
test_egsm_up2 = read_data(31,890,895).reshape(31,201)
#############egsm_down#########4
test_egsm_down1 = read_data(31,903,908).reshape(31,201)
test_egsm_down2 = read_data(31,908,913).reshape(31,201)
############satellite##########
##############wlan##############5
test_wlan1 = read_data(31,2400,2405).reshape(31,201)
test_wlan2 = read_data(31,2405,2410).reshape(31,201)
##############4G################6
test_lte1 = read_data(31,1880,1885).reshape(31,201)
test_lte2 = read_data(31,1885,1890).reshape(31,201)
##############3G##################7
test_scdma1 = read_data(31,2010.5,2015.5).reshape(31,201)
test_scdma2 = read_data(31,2015.5,2020.5).reshape(31,201)
#######
test_cdma_down1 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_cdma_down1] )
test_cdma_down2 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_cdma_down2] )
test_cdma_up1 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_cdma_up1] )
test_cdma_up2 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_cdma_up2] )
test_lte1 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_lte1] )
test_lte2 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_lte2] )
test_scdma1 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_scdma1] )
test_scdma2 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_scdma2] )
test_wlan1 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_wlan1] )
test_wlan2 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_wlan2] )
test_egsm_up1 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_egsm_up1] )
test_egsm_up2 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_egsm_up2] )
test_egsm_down1 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_egsm_down1] )
test_egsm_down2 = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_egsm_down2] )
##############
test_data = [test_cdma_up1,test_cdma_up2,test_cdma_down1,test_cdma_down2,
                   test_egsm_up1,test_egsm_up2,test_egsm_down1,
                        test_egsm_down2,test_wlan1,test_wlan2,
                        test_lte1,test_lte2,test_scdma1,test_scdma2]

test_data = [test_egsm_up1,test_egsm_up2,test_egsm_down1,
                        test_egsm_down2]
test = test_data[0]
for i in range(len(test_data)-1):
    test = np.vstack((test,test_data[i+1]))
#######################################shuffle#######
idx=random.sample(range(3400),3400) 
idx1 = random.sample(range(124),124)
train = train[idx]
train=train.astype(np.float32)
test = test[idx1]
test = test.astype(np.float32)



##############################################model 共性###############
dropout=0.5
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 200],name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 2],name='y-input')
with tf.name_scope('input_reshape'):
    x_image = tf.reshape(x,[-1,1,200,1])
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
with tf.name_scope('maxpool_3'):
    h_pool3 = max_pool_2x2(h_conv2)
    tf.summary.histogram('h_pool3', h_pool3)
with tf.name_scope('conv_4'):
    with tf.name_scope('weights'):
        W_conv4 = weight_variable([1, 3, 64, 128])
        variable_summaries(W_conv4)
    with tf.name_scope('biases'):
        b_conv4 = bias_variable([128])
        variable_summaries(b_conv4)
    with tf.name_scope('conv'):
        conv4 = conv2d(h_pool3, W_conv4)
        tf.summary.histogram('conv',conv4)
    h_conv4 = tf.nn.relu(conv4+b_conv4,name='activation')
    tf.summary.histogram('h_conv4', h_conv4)
with tf.name_scope('maxpool_5'):
    h_pool5 = max_pool_2x2(h_conv4)
    tf.summary.histogram('h_pool5', h_pool5)
h_pool6_flat = tf.reshape(h_pool5, [-1, 1*50*128])
with tf.name_scope('fc_1'):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      W_fc1 = weight_variable([1* 50 * 128, 256])
      variable_summaries(W_fc1)
    with tf.name_scope('biases'):
      b_fc1 = bias_variable([256])
      variable_summaries(b_fc1)
    with tf.name_scope('Wx_plus_b'):
      preactivate1 = tf.matmul(h_pool6_flat, W_fc1) + b_fc1
      tf.summary.histogram('pre_activations', preactivate1)
    activations1 = tf.nn.relu(preactivate1, name='activation')
    tf.summary.histogram('activations', activations1)
with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)
  tf.summary.scalar('dropout_keep_probability', keep_prob)
  h_fc1_drop = tf.nn.dropout(activations1, keep_prob)
with tf.name_scope('fc_2'):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      W_fc2 = weight_variable([256,2])
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
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(activations2, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)


##############################################






##############model1#######################################
log_dir1 = 'D:/temp/tensorflow/project/logs/projectcnn_with_summaries1'
enc = preprocessing.OneHotEncoder()  
enc.fit([[1],[0]])  

array1_1 = enc.transform([[1]]*1700).toarray()  
array1_2 = enc.transform([[0]]*10200).toarray()  
label1 = np.vstack((array1_1,array1_2))
array1_1_test = enc.transform([[1]]*62).toarray()
array1_2_test = enc.transform([[0]]*372).toarray()
label1_test = np.vstack((array1_1_test,array1_2_test))
# 顺序打乱
label1 = label1[idx]
label1 = label1.astype(np.float32)
label1_test = label1_test.astype(np.float32)
#set up model
sess1  = tf.InteractiveSession()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir1 + '/train', sess1.graph)
test_writer = tf.summary.FileWriter(log_dir1 + '/test')
tf.global_variables_initializer().run()
"""
def feed_dict1(train,a):
  """""""Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""""""
  if train:
    xs = train[a*50:a*50+50]
    ys = label1[a*50:a*50+50]
    k = 0.5
  else:
    xs = test
    ys = label1_test
    k = 0.999
  return {x: xs, y_: ys, keep_prob: k}
"""

saver = tf.train.Saver()  
for i in range(238):
    if i % 10 == 0:  # Record summaries and test-set accuracy
        summary, acc = sess1.run([merged, accuracy], feed_dict={x:test,y_:label1_test,keep_prob:1.0})
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
    else:
        # Record train set summaries, and train
        if i % 100 == 99:  # Record execution stats
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess1.run([merged, train_step],
                            feed_dict={x:train[a*50:a*50+50],y_:label1[a*50:a*50+50],keep_prob:0.5},
                            options=run_options,
                            run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            saver.save(sess1, log_dir1+"/model.ckpt", i)
            print('Adding run metadata for', i)
        else:  # Record a summary
            summary, _ = sess1.run([merged, train_step], feed_dict={x:train[a*50:a*50+50],y_:label1[a*50:a*50+50],keep_prob:0.5})
            train_writer.add_summary(summary, i)
            print(2)
train_writer.close()
test_writer.close()

###################model2######################################################
log_dir2 = 'D:/temp/tensorflow/project/logs/projectcnn_with_summaries2'
enc = preprocessing.OneHotEncoder()  
enc.fit([[1],[0]])  

array2_1 = enc.transform([[0]]*1700).toarray()
array2_2 = enc.transform([[1]]*1700).toarray()  
array2_3 = enc.transform([[0]]*8500).toarray()  
label2 = np.vstack((array2_1,array2_2,array2_3))
array2_1_test = enc.transform([[0]]*62).toarray()
array2_2_test = enc.transform([[1]]*62).toarray()  
array2_3_test = enc.transform([[0]]*310).toarray()  
label2_test = np.vstack((array2_1_test,array2_2_test,array2_3_test))
# 顺序打乱
label2 = label1[idx]
label2 = label1.astype(np.float32)
sess2 = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir2 + '/train', sess2.graph)
test_writer = tf.summary.FileWriter(log_dir2 + '/test')
tf.global_variables_initializer().run()
def feed_dict2(train,a):
  """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
  if train:
    xs = train[a-1:a]
    ys = label2[a-1:a]
    k = dropout
  else:
    xs = test
    ys = label2_test
    k = 1.0
  return {x: xs, y_: ys, keep_prob: k}
##########################model3##################################################
log_dir3 = 'D:/temp/tensorflow/project/logs/projectcnn_with_summaries3'
enc = preprocessing.OneHotEncoder()  
enc.fit([[1],[0]])  

array3_1 = enc.transform([[0]]*3400).toarray()
array3_2 = enc.transform([[1]]*1700).toarray()  
array3_3 = enc.transform([[0]]*6800).toarray()  
label3 = np.vstack((array3_1,array3_2,array3_3))
array3_1_test = enc.transform([[0]]*124).toarray()
array3_2_test = enc.transform([[1]]*62).toarray()  
array3_3_test = enc.transform([[0]]*248).toarray()  
label3_test = np.vstack((array3_1_test,array3_2_test,array3_3_test))
# 顺序打乱
label3 = label1[idx]
label3 = label1.astype(np.float32)
sess3 = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir3 + '/train', sess3.graph)
test_writer = tf.summary.FileWriter(log_dir3 + '/test')
tf.global_variables_initializer().run()
def feed_dict3(train,a):
  """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
  if train:
    xs = train[a-1:a]
    ys = label3[a-1:a]
    k = dropout
  else:
    xs = test
    ys = label3_test
    k = 1.0
  return {x: xs, y_: ys, keep_prob: k}
###############################model4###########################################
log_dir4 = 'D:/temp/tensorflow/project/logs/projectcnn_with_summaries4'
enc = preprocessing.OneHotEncoder()  
enc.fit([[1],[0]])  

array4_1 = enc.transform([[0]]*5100).toarray()
array4_2 = enc.transform([[1]]*1700).toarray()  
array4_3 = enc.transform([[0]]*5100).toarray()  
label4 = np.vstack((array4_1,array4_2,array4_3))
array4_1_test = enc.transform([[0]]*186).toarray()
array4_2_test = enc.transform([[1]]*62).toarray()  
array4_3_test = enc.transform([[0]]*186).toarray()  
label4_test = np.vstack((array4_1_test,array4_2_test,array4_3_test))
# 顺序打乱
label4 = label1[idx]
label4 = label1.astype(np.float32)
sess4 = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir4 + '/train', sess4.graph)
test_writer = tf.summary.FileWriter(log_dir4 + '/test')
tf.global_variables_initializer().run()
def feed_dict4(train,a):
  """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
  if train:
    xs = train[a-1:a]
    ys = label4[a-1:a]
    k = dropout
  else:
    xs = test
    ys = label4_test
    k = 1.0
  return {x: xs, y_: ys, keep_prob: k}
############################################model5#################################
log_dir5 = 'D:/temp/tensorflow/project/logs/projectcnn_with_summaries5'
enc = preprocessing.OneHotEncoder()  
enc.fit([[1],[0]])  

array5_1 = enc.transform([[0]]*6800).toarray()
array5_2 = enc.transform([[1]]*1700).toarray()  
array5_3 = enc.transform([[0]]*3400).toarray()  
label5 = np.vstack((array5_1,array5_2,array5_3))
array5_1_test = enc.transform([[0]]*248).toarray()
array5_2_test = enc.transform([[1]]*62).toarray()  
array5_3_test = enc.transform([[0]]*124).toarray()  
label5_test = np.vstack((array5_1_test,array5_2_test,array5_3_test))
# 顺序打乱
label5 = label1[idx]
label5 = label1.astype(np.float32)
sess5 = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir5 + '/train', sess5.graph)
test_writer = tf.summary.FileWriter(log_dir5 + '/test')
tf.global_variables_initializer().run()
def feed_dict5(train,a):
  """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
  if train:
    xs = train[a-1:a]
    ys = label5[a-1:a]
    k = dropout
  else:
    xs = test
    ys = label5_test
    k = 1.0
  return {x: xs, y_: ys, keep_prob: k}
#################################################model6############################
log_dir6 = 'D:/temp/tensorflow/project/logs/projectcnn_with_summaries6'
enc = preprocessing.OneHotEncoder()  
enc.fit([[1],[0]])  

array6_1 = enc.transform([[0]]*8500).toarray()
array6_2 = enc.transform([[1]]*1700).toarray()  
array6_3 = enc.transform([[0]]*1700).toarray()  
label6 = np.vstack((array6_1,array6_2,array6_3))
array6_1_test = enc.transform([[0]]*310).toarray()
array6_2_test = enc.transform([[1]]*62).toarray()  
array6_3_test = enc.transform([[0]]*62).toarray()  
label6_test = np.vstack((array6_1_test,array6_2_test,array6_3_test))
# 顺序打乱
label6 = label1[idx]
label6 = label1.astype(np.float32)
sess6 = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir6 + '/train', sess6.graph)
test_writer = tf.summary.FileWriter(log_dir6 + '/test')
tf.global_variables_initializer().run()
def feed_dict6(train,a):
  """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
  if train:
    xs = train[a-1:a]
    ys = label6[a-1:a]
    k = dropout
  else:
    xs = test
    ys = label6_test
    k = 1.0
  return {x: xs, y_: ys, keep_prob: k}
##########################################model7######################################
log_dir7 = 'D:/temp/tensorflow/project/logs/projectcnn_with_summaries7'
enc = preprocessing.OneHotEncoder()  
enc.fit([[1],[0]])  

array7_1 = enc.transform([[0]]*10200).toarray()
array7_2 = enc.transform([[1]]*1700).toarray()  
label7 = np.vstack((array7_1,array7_2))
array7_1_test = enc.transform([[0]]*372).toarray()
array7_2_test = enc.transform([[1]]*62).toarray()  
label7_test = np.vstack((array7_1_test,array7_2_test))
# 顺序打乱
label1 = label1[idx]
label1 = label1.astype(np.float32)
sess7 = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir7 + '/train', sess7.graph)
test_writer = tf.summary.FileWriter(log_dir7 + '/test')
tf.global_variables_initializer().run()
def feed_dict7(train,a):
  """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
  if train:
    xs = train[a-1:a]
    ys = label7[a-1:a]
    k = dropout
  else:
    xs = test
    ys = label7_test
    k = 1.0
  return {x: xs, y_: ys, keep_prob: k}




