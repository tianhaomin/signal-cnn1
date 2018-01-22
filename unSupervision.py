# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:51:11 2018

@author: Administrator
"""
from sklearn.manifold import TSNE
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import os
from keras import optimizers
sns.set(style="ticks")
sns.set(style="whitegrid", palette="pastel", color_codes=True)
#load data
train = np.load("F://tmp//data//raw data//train.npy")
label1=np.load("F://tmp//data//raw data//label_train.npy")
test=np.load("F://tmp//data//raw data//test.npy")
label1_test=np.load("F://tmp//data//raw data//label_test.npy")
#set up cnn mmodel
sess1  = tf.InteractiveSession()
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 200],name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 4],name='y-input')
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
    for j in range(85):
        if j % 10 == 0:  # Record summaries and test-set accuracy
            acc = sess1.run([accuracy], feed_dict={x:test,y_:label1_test,keep_prob:1.0})
            result_test.append(acc)
            print('Accuracy at step %s of epoch %s: %s' % (j,i, acc))
        else:  # Record a summary
    #         t1=time.time()
            _ = sess1.run([train_step], feed_dict={x:train[50*j:50*(j+1)],y_:label1[50*j:50*(j+1)],keep_prob:0.8})
            acc_train = sess1.run([accuracy],feed_dict={x:train[0:200],y_:label1[0:200],keep_prob:0.8})
            result_train.append(acc_train)
            print(acc_train)
                #print(2)
    #         t2=time.time()
    #         print(t2-t1)

sess1.close()

###t-sne可视化特征
feature = sess1.run([activations1],feed_dict={x:test,y_:label1_test,keep_prob:1.0})
feature = feature[0]
X_embedded = TSNE(n_components=2).fit_transform(feature)
colors=['red', 'green', 'blue','yellow']
plt.scatter(X_embedded[:,0],X_embedded[:,1])
plt.show()

plt.scatter(test[:,0],test[:,1])
plt.show()

############################################################################set up autoencoder###################################################
input_x = Input(shape=(1,200,1))  # tensorflow后端
x = Conv2D(32, (1,1),activation='relu', padding='same')(input_x)
#x = Conv2D(48, (1,3), activation='relu', padding='same')(x)
#x = MaxPooling2D(pool_size=(2,2),strides=(2,2), padding='same')(x)
x = Conv2D(64, (1,1),strides=2, activation='relu', padding='same')(x)
#x = Conv2D(80, (1,3), activation='relu', padding='same')(x)
#x = MaxPooling2D(pool_size=(2,2),strides=(2,2), padding='same')(x)
#x = Conv2D(128, (1,3), activation='relu', padding='same')(x)
x = Conv2D(156, (1,3), activation='relu', padding='same')(x)
#x = MaxPooling2D(pool_size=(2,2),strides=(2,2), padding='same')(x)
x = Conv2D(200, (1,3),strides=2, activation='relu', padding='same')(x)
#x = Conv2D(256, (1,3), activation='relu', padding='same')(x)
#x = Conv2D(300, (1,3), activation='relu', padding='same')(x)

encoded = MaxPooling2D(pool_size=(2,2),strides=(2,2), padding='same')(x)
# 解码器
#x = UpSampling2D((1, 2))(encoded)
#x = Conv2D(300, (1,3), activation='relu', padding='same')(x)
#x = Conv2D(256, (1,3), activation='relu', padding='same')(x)
x = Conv2D(200, (1,3), activation='relu', padding='same')(x)
#x = UpSampling2D((1,2))(x)
x = Conv2D(156, (1,3), activation='relu', padding='same')(x)
#x = Conv2D(128, (1,3), activation='relu', padding='same')(x)
x = UpSampling2D((1,2))(x)
#x = Conv2D(80, (1,3), activation='relu', padding='same')(x)
x = Conv2D(64, (1,1), activation='relu', padding='same')(x)
x = UpSampling2D((1,2))(x)
#x = Conv2D(48, (1,3), activation='relu', padding='same')(x)
x = Conv2D(32, (1,1), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 1))(x)

decoded = Conv2D(1, (1, 3), activation='relu', padding='same')(x)

autoencoder = Model(input_x, decoded)
encoder = Model(inputs=input_x, outputs=encoded) 
autoencoder.compile(optimizer='Adam', loss='mse')  
train = train.reshape(3400,1,200,1)
autoencoder.fit(train, train, epochs=500, batch_size=56, shuffle=True)  
test = test.reshape(124,1,200,1)      
results = encoder.predict(test)
results = results.reshape(124,5000)
X_embedded = TSNE(n_components=2).fit_transform(results)
plt.scatter(X_embedded[:,0],X_embedded[:,1])
plt.show()



