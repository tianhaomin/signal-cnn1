# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 15:33:56 2017

@author: Administrator
"""
#downlink有tensorboard版本自创的深层基本结构
#############good
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
        df = pandas.read_table("F:/project/Yin/spectrum-data//"+a[i],names=["fc","E"])
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


log_dir1 = 'D:/temp/tensorflow/project/logs/zuizhong'

sess1  = tf.InteractiveSession()
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 200],name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 4],name='y-input')
with tf.name_scope('input_reshape'):
    x_image = tf.reshape(x,[-1,1,200,1])
    tf.summary.image('input', x_image, 4)
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
        W_conv2 = weight_variable([1, 3, 32, 48])
        variable_summaries(W_conv2)
    with tf.name_scope('biases'):
        b_conv2 = bias_variable([48])
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
        W_conv4 = weight_variable([1, 1, 48, 64])
        variable_summaries(W_conv4)
    with tf.name_scope('biases'):
        b_conv4 = bias_variable([64])
        variable_summaries(b_conv4)
    with tf.name_scope('conv'):
        conv4 = conv2d(h_pool3, W_conv4)
        tf.summary.histogram('conv',conv4)
    h_conv4 = tf.nn.relu(conv4+b_conv4,name='activation')
    tf.summary.histogram('h_conv4', h_conv4)
with tf.name_scope('conv_5'):
    with tf.name_scope('weights'):
        W_conv5 = weight_variable([1, 3, 64, 80])
        variable_summaries(W_conv5)
    with tf.name_scope('biases'):
        b_conv5 = bias_variable([80])
        variable_summaries(b_conv5)
    with tf.name_scope('conv'):
        conv5 = conv2d(h_conv4, W_conv5)
        tf.summary.histogram('conv',conv5)
    h_conv5 = tf.nn.relu(conv5+b_conv5,name='activation')
    tf.summary.histogram('h_conv5', h_conv5)
with tf.name_scope('maxpool_5'):
    h_pool5 = max_pool_2x2(h_conv5)
    tf.summary.histogram('h_pool5', h_pool5)
with tf.name_scope('conv_6'):
    with tf.name_scope('weights'):
        W_conv6 = weight_variable([1, 3, 80, 128])
        variable_summaries(W_conv6)
    with tf.name_scope('biases'):
        b_conv6 = bias_variable([128])
        variable_summaries(b_conv6)
    with tf.name_scope('conv'):
        conv6 = conv2d(h_pool5, W_conv6)
        tf.summary.histogram('conv',conv6)
    h_conv6 = tf.nn.relu(conv6+b_conv6,name='activation')
    tf.summary.histogram('h_conv6', h_conv6)
with tf.name_scope('conv_7'):
    with tf.name_scope('weights'):
        W_conv7 = weight_variable([1, 3, 128, 156])
        variable_summaries(W_conv7)
    with tf.name_scope('biases'):
        b_conv7 = bias_variable([156])
        variable_summaries(b_conv7)
    with tf.name_scope('conv'):
        conv7 = conv2d(h_conv6, W_conv7)
        tf.summary.histogram('conv',conv7)
    h_conv7 = tf.nn.relu(conv7+b_conv7,name='activation')
    tf.summary.histogram('h_conv7', h_conv7)
with tf.name_scope('maxpool_8'):
    h_pool8 = max_pool_2x2(h_conv7)
    tf.summary.histogram('h_pool8', h_pool8)
with tf.name_scope('conv_9'):
    with tf.name_scope('weights'):
        W_conv9 = weight_variable([1, 3, 156, 200])
        variable_summaries(W_conv9)
    with tf.name_scope('biases'):
        b_conv9 = bias_variable([200])
        variable_summaries(b_conv9)
    with tf.name_scope('conv'):
        conv9 = conv2d(h_pool8, W_conv9)
        tf.summary.histogram('conv',conv9)
    h_conv9 = tf.nn.relu(conv9+b_conv9,name='activation')
    tf.summary.histogram('h_conv9', h_conv9)
with tf.name_scope('conv_10'):
    with tf.name_scope('weights'):
        W_conv10 = weight_variable([1, 3, 200, 256])
        variable_summaries(W_conv10)
    with tf.name_scope('biases'):
        b_conv10 = bias_variable([256])
        variable_summaries(b_conv10)
    with tf.name_scope('conv'):
        conv10 = conv2d(h_conv9, W_conv10)
        tf.summary.histogram('conv',conv10)
    h_conv10 = tf.nn.relu(conv10+b_conv10,name='activation')
    tf.summary.histogram('h_conv10', h_conv10)
with tf.name_scope('conv_11'):
    with tf.name_scope('weights'):
        W_conv11 = weight_variable([1, 3, 256, 300])
        variable_summaries(W_conv11)
    with tf.name_scope('biases'):
        b_conv11 = bias_variable([300])
        variable_summaries(b_conv11)
    with tf.name_scope('conv'):
        conv11 = conv2d(h_conv10, W_conv11)
        tf.summary.histogram('conv',conv11)
    h_conv11 = tf.nn.relu(conv11+b_conv11,name='activation')
with tf.name_scope('maxpool_12'):
    h_pool12 = max_pool_2x2(h_conv11)
    tf.summary.histogram('h_pool12', h_pool12)
h_pool12_flat = tf.reshape(h_pool12, [-1, 1*13*300])
with tf.name_scope('fc_1'):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
        W_fc1 = weight_variable([1* 13 * 300, 5000])
        variable_summaries(W_fc1)
    with tf.name_scope('biases'):
        b_fc1 = bias_variable([5000])
        variable_summaries(b_fc1)
    with tf.name_scope('Wx_plus_b'):
        preactivate1 = tf.matmul(h_pool12_flat, W_fc1) + b_fc1
        tf.summary.histogram('pre_activations', preactivate1)
    activations1 = tf.nn.relu(preactivate1, name='activation')
    tf.summary.histogram('activations', activations1)
#keep_prob = tf.placeholder(tf.float32)
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32,name='dropout')
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    h_fc1_drop = tf.nn.dropout(activations1, keep_prob)
with tf.name_scope('fc_2'):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
        W_fc2 = weight_variable([5000,4])
        variable_summaries(W_fc2)
    with tf.name_scope('biases'):
        b_fc2 = bias_variable([4])
        variable_summaries(b_fc2)
    with tf.name_scope('Wx_plus_b'):
        preactivate2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        tf.summary.histogram('pre_activations', preactivate2)
    activations2 = tf.nn.softmax(preactivate2, name='activation')
    tf.summary.histogram('activations', activations2)
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(activations2,1e-10,1.0)), reduction_indices=[1]))
tf.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
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
for i in range(10000):
    a = random.randint(0,3200)
    if i % 10 == 0:  # Record summaries and test-set accuracy
        summary,acc = sess1.run([merged,accuracy], feed_dict={x:test,y_:label1_test,keep_prob:1.0})
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
    else: 
         if i % 100 == 99:
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
         else:
             summary,_ = sess1.run([summary,train_step], feed_dict={x:train[a:a+10],y_:label1[a:a+10],keep_prob:0.8})
             train_writer.add_summary(summary, i)

sess1.close()




#卷积层输出可视化
xx=test[0].reshape(1,200)
yy=label1_test[-1].reshape(1,4)
layer_conv1 = sess1.run([h_conv1],feed_dict={x:xx,y_:yy,keep_prob:1.0})
layer_conv1 = layer_conv1[0].reshape(1,200,32)


fig, axes = plt.subplots(nrows=8, ncols=4, figsize=(12, 12))
for row in range(8):
    for col in range(4):
        ot=layer_conv1[:,:,4*row+col]
        x_temp=[]
        y_temp=[]
        for k in range(200):
            if ot[0][k]>0:
                axes[row,col].plot([k,k],[0,ot[0][k]])
                axes[row,col].scatter(k,ot[0][k])
fig.tight_layout()
plt.show()        

layer_conv11 = sess1.run([h_conv11],feed_dict={x:xx,y_:yy,keep_prob:1.0})
layer_conv11 = layer_conv11[0].reshape(1,25,300)

fig, axes = plt.subplots(nrows=10, ncols=4, figsize=(12, 12))
for row in range(10):
    for col in range(4):
        ot=layer_conv11[:,:,4*row+col]
        for k in range(25):
            if ot[0][k]>0:
                axes[row,col].plot([k,k],[0,ot[0][k]])
                axes[row,col].scatter(k,ot[0][k])
fig.tight_layout()
plt.show()        







