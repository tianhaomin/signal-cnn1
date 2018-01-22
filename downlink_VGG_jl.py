# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 21:53:42 2017

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

import tensorflow as tf
log_dir1 = 'D:/temp/tensorflow/project/logs/VGG_downlink'

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

def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[kh, kw, n_in, n_out],  #kh、kw-卷积核尺寸，n_out卷积核数目
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        variable_summaries(kernel)
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        variable_summaries(biases)
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation

def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[n_in, n_out],
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer())
        variable_summaries(kernel)
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        variable_summaries(biases)
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation

def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)


def inference_op(input_op, keep_prob):
    p = []
    # assume input_op shape is 224x224x3

    # block 1 -- outputs 112x112x64
    conv1_1 = conv_op(input_op, name="conv1_1", kh=1, kw=3, n_out=64, dh=1, dw=1, p=p)
    tf.summary.histogram('conv1_1',conv1_1)
    conv1_2 = conv_op(conv1_1,  name="conv1_2", kh=1, kw=3, n_out=64, dh=1, dw=1, p=p)
    tf.summary.histogram('conv1_2',conv1_2)
    pool1 = mpool_op(conv1_2,   name="pool1",   kh=1, kw=2, dw=2, dh=2)
    tf.summary.histogram('pool1',pool1)

    # block 2 -- outputs 56x56x128
    conv2_1 = conv_op(pool1,    name="conv2_1", kh=1, kw=3, n_out=128, dh=1, dw=1, p=p)
    tf.summary.histogram('conv2_1',conv2_1)
    conv2_2 = conv_op(conv2_1,  name="conv2_2", kh=1, kw=3, n_out=128, dh=1, dw=1, p=p)
    tf.summary.histogram('conv2_2',conv2_2)
    pool2 = mpool_op(conv2_2,   name="pool2",   kh=1, kw=2, dh=2, dw=2)
    tf.summary.histogram('pool2',pool2)

    # # block 3 -- outputs 28x28x256
    conv3_1 = conv_op(pool2,    name="conv3_1", kh=1, kw=3, n_out=256, dh=1, dw=1, p=p)
    tf.summary.histogram('conv3_1',conv3_1)
    conv3_2 = conv_op(conv3_1,  name="conv3_2", kh=1, kw=3, n_out=256, dh=1, dw=1, p=p)
    tf.summary.histogram('conv3_2',conv3_2)
    conv3_3 = conv_op(conv3_2,  name="conv3_3", kh=1, kw=3, n_out=256, dh=1, dw=1, p=p)    
    tf.summary.histogram('conv3_3',conv3_3)
    pool3 = mpool_op(conv3_3,   name="pool3",   kh=1, kw=2, dh=2, dw=2)
    tf.summary.histogram('pool3',pool3)

    # block 4 -- outputs 14x14x512
    conv4_1 = conv_op(pool3,    name="conv4_1", kh=1, kw=3, n_out=512, dh=1, dw=1, p=p)
    tf.summary.histogram('conv4_1',conv4_1)
    conv4_2 = conv_op(conv4_1,  name="conv4_2", kh=1, kw=3, n_out=512, dh=1, dw=1, p=p)
    tf.summary.histogram('conv4_2',conv4_2)
    conv4_3 = conv_op(conv4_2,  name="conv4_3", kh=1, kw=3, n_out=512, dh=1, dw=1, p=p)
    tf.summary.histogram('conv4_3',conv4_3)
    pool4 = mpool_op(conv4_3,   name="pool4",   kh=1, kw=2, dh=2, dw=2)
    tf.summary.histogram('pool4',pool4)

    # block 5 -- outputs 7x7x512
    conv5_1 = conv_op(pool4,    name="conv5_1", kh=1, kw=3, n_out=512, dh=1, dw=1, p=p)
    tf.summary.histogram('conv5_1',conv5_1)
    conv5_2 = conv_op(conv5_1,  name="conv5_2", kh=1, kw=3, n_out=512, dh=1, dw=1, p=p)
    tf.summary.histogram('conv5_2',conv5_2)
    conv5_3 = conv_op(conv5_2,  name="conv5_3", kh=1, kw=3, n_out=512, dh=1, dw=1, p=p)
    tf.summary.histogram('conv5_3',conv5_3)
    pool5 = mpool_op(conv5_3,   name="pool5",   kh=1, kw=2, dw=2, dh=2)
    tf.summary.histogram('pool5',pool5)

    # flatten
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    # fully connected
    fc6 = fc_op(resh1, name="fc6", n_out=1024, p=p)
    tf.summary.histogram('fc6',fc6)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    fc7 = fc_op(fc6_drop, name="fc7", n_out=1024, p=p)
    tf.summary.histogram('fc7',fc7)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    fc8 = fc_op(fc7_drop, name="fc8", n_out=4, p=p)
    tf.summary.histogram('fc8',fc8)
    softmax = tf.nn.softmax(fc8)
    tf.summary.histogram('softmax',softmax)
    predictions = tf.argmax(softmax, 1)
    
    return predictions, softmax, fc8, p


sess1  = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 200],name='x-input')
y_ = tf.placeholder(tf.float32, [None, 4],name='y-input')
x_image = tf.reshape(x,[-1,1,200,1])
keep_prob = tf.placeholder(tf.float32,name='dropout')    
predictions,softmax,fc_result,p = inference_op(x_image, keep_prob)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(softmax,1e-10,1.0)), reduction_indices=[1]))
tf.summary.scalar('cross_entropy', cross_entropy)
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(predictions, tf.argmax(y_, 1))
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




