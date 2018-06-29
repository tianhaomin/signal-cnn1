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
data_cdma_down = read_data(850,870,875).reshape(850,201)
#############egsm_down#########4
data_egsm_down = read_data(850,930,935).reshape(850,201)
data_wcdma_down = read_data(850,2135,2140).reshape(850,201)
##############4G################6
data_lte_down = read_data(850,1850,1855).reshape(850,201)
#data_evdo_down = read_data(850,1920,1930).reshape(850,401)
data_dcs_down = read_data(850,1805,1810).reshape(850,201)
data_tv = read_data(850,100,105).reshape(850,201)

data_cdma_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_cdma_down] )
data_egsm_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_egsm_down] )
data_wcdma_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_wcdma_down] )
data_lte_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_lte_down] )
#data_evdo_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_evdo_down] )
data_dcs_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_dcs_down] )
data_tv = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_tv] )

train_data = [data_cdma_down,data_egsm_down,data_wcdma_down,data_lte_down,data_dcs_down,data_tv]
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
test_cdma_down = read_data(31,870,875).reshape(31,201)
#############egsm_down#########4
test_egsm_down = read_data(31,930,935).reshape(31,201)
##########lte下行#########
test_lte_down = read_data(31,1850,1855).reshape(31,201)
#########wcdma下行###########
test_wcdma_down = read_data(31,2135,2140).reshape(31,201)
#test_evdo_down = read_data(31,1920,1930).reshape(31,401)
test_dcs_down = read_data(31,1805,1810).reshape(31,201)
test_tv = read_data(31,100,105).reshape(31,201)

#######
test_cdma_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_cdma_down] )
test_egsm_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_egsm_down] )
test_lte_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_lte_down] )
test_wcdma_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_wcdma_down] )
#test_evdo_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_evdo_down] )
test_dcs_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_dcs_down] )
test_tv = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_tv] )
##############
test_data = [test_cdma_down,test_egsm_down,test_wcdma_down,test_lte_down,test_dcs_down,test_tv]

test = test_data[0]
for i in range(len(test_data)-1):
    test = np.vstack((test,test_data[i+1]))
#######################################shuffle#######
train=train.astype(np.float32)
test = test.astype(np.float32)
enc = preprocessing.OneHotEncoder()  
enc.fit([[1],[2],[3],[4],[5],[6]])  

array1_1 = enc.transform([[1]]*850).toarray()  
array1_2 = enc.transform([[2]]*850).toarray()  
array1_3 = enc.transform([[3]]*850).toarray()  
array1_4 = enc.transform([[4]]*850).toarray()  
array1_5 = enc.transform([[5]]*850).toarray()  
array1_6 = enc.transform([[6]]*850).toarray()  
label1 = np.vstack((array1_1,array1_2,array1_3,array1_4,array1_5,array1_6))
array1_1_test = enc.transform([[1]]*31).toarray()
array1_2_test = enc.transform([[2]]*31).toarray()
array1_3_test = enc.transform([[3]]*31).toarray()
array1_4_test = enc.transform([[4]]*31).toarray()
array1_5_test = enc.transform([[5]]*31).toarray()
array1_6_test = enc.transform([[6]]*31).toarray()
label1_test = np.vstack((array1_1_test,array1_2_test,array1_3_test,array1_4_test,array1_5_test,array1_6_test))
# 顺序打乱
label1 = label1.astype(np.float32)
label1_test = label1_test.astype(np.float32)


idx=random.sample(range(5100),5100) 
idx1 = random.sample(range(186),186)
train = train[idx]
test = test[idx1]
label1 = label1[idx]
label1_test = label1_test[idx1]

#train = np.load("F://tmp//data//raw data//train.npy")
#label1=np.load("F://tmp//data//raw data//label_train.npy")
#test=np.load("F://tmp//data//raw data//test.npy")
#label1_test=np.load("F://tmp//data//raw data//label_test.npy")
##
#
#for i in train:
#    idx=random.sample(range(200),180) 
#    i[idx] = 0
#
#for i in test:
#    idx=random.sample(range(200),180) 
#    i[idx] = 0
#    
sess1  = tf.InteractiveSession()
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 200],name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 6],name='y-input')
with tf.name_scope('input_reshape'):
    x_image = tf.reshape(x,[-1,1,200,1])
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
with tf.name_scope('maxpool_12'):
    h_pool12 = max_pool_2x2(h_conv11)
h_pool12_flat = tf.reshape(h_pool12, [-1, 1*13*300])
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
        W_fc2 = weight_variable([5000,6])
    with tf.name_scope('biases'):
        b_fc2 = bias_variable([6])
    with tf.name_scope('Wx_plus_b'):
        preactivate2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
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
#tf.summary.scalar('accuracy', accuracy)



#log_dir1="F://tmp//tflearn_logs//our"
#merged = tf.summary.merge_all()
#train_writer = tf.summary.FileWriter(log_dir1 + '//train', sess1.graph)
#test_writer = tf.summary.FileWriter(log_dir1 + '//test')
#idx=random.sample(range(4250),4250) 
#train = train[idx]
#label1 = label1[idx]
tf.global_variables_initializer().run()


#    a = random.randint(0,4199)
    
#    else:
#        # Record train set summaries, and train
#        if i % 100 == 99:  # Record execution stats
#            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#            run_metadata = tf.RunMetadata()
#            summary, _ = sess1.run([merged, train_step],
#                            feed_dict={x:train[a:a+128],y_:label1[a:a+128],keep_prob:0.8},
#                            options=run_options,
#                            run_metadata=run_metadata)
#            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
#            train_writer.add_summary(summary, i)
#            saver.save(sess1, log_dir1+"/model.ckpt", i)
#            print('Adding run metadata for', i)
result_test=[]
result_train=[]
for i in range(3000):
    for j in range(102):
        if j % 10 == 0:  # Record summaries and test-set accuracy
            acc = sess1.run([accuracy], feed_dict={x:test,y_:label1_test,keep_prob:1.0})
            result_test.append(acc)
            print('Accuracy at step %s of epoch %s: %s' % (j,i, acc))
        else:  # Record a summary
    #         t1=time.time()
            _ = sess1.run([train_step], feed_dict={x:train[50*j:50*(j+1)],y_:label1[50*j:50*(j+1)],keep_prob:0.8})
            acc_train = sess1.run([accuracy],feed_dict={x:train[50*j:50*(j+1)],y_:label1[50*j:50*(j+1)],keep_prob:0.8})
            result_train.append(acc_train)
            print(acc_train)
                #print(2)
    #         t2=time.time()
    #         print(t2-t1)

sess1.close()




save1=np.array(result_train)
np.save("F://tmp//data//add new signal//1x1+1x5_train.npy",result_train)
np.save("F://tmp//data//add new signal//1x1+1x5_valid.npy",result_test)

import seaborn as sns
sns.set()
filter1 = sess1.run(W_conv4)
filter2 = sess1.run(W_conv2)
y_ = filter2[:,:,0,0]
plt.plot([1,1],[0,y_[0][0]])
plt.show()
for i in range(32):
    y_ = filter2[:,:,i,0]
    for j in range(3):
        plt.plot([j+1,j+1],[0,y_[0][j]])
    plt.show()
    
    
fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(12, 12))
for row in range(8):
    for col in range(8):
        x = [j for j in range(48)]
        x = np.array(x)
        y = [k for k in filter1[:,:,:,(8*row+col)]]
        y = np.array(y).reshape(48,)
        axes[row,col].plot(x,y)
    
fig.tight_layout()
plt.show()
    