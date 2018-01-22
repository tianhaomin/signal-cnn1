# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:29:42 2017

@author: Administrator
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#导入数据创建爱你交互模式
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
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
                        strides=[1, 2, 2, 1], padding='SAME')  
#规定输入输出的格式，因为原先是得到784维的向量而卷机网络是处理
#二维的图像所以将向量转为28*28的格式                       
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1,28,28,1])
#定义第一个卷积层                       
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
#权重和偏置设置好就开始做卷积处理conv2d(x,w)然后进行RELU非线性激活处理
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#最大池化处理
h_pool1 = max_pool_2x2(h_conv1)
#同上第二个卷积层的定义，前一层有32个卷积核这一层的通道数就有32
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#因为前面经历了两次的2*2 的最大池化反应，所以边长只有1/4图片尺寸变成了7*7而
#第二个卷积核的数量有64个所以输出的tensor的尺寸是7*7*64
#我们需要将输出的tensor进行变形整合成一维向量然后连接一个全连接
#隐藏节点是1024并且使用RELU进行激活
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#用dropout层减少过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#用softmax实现多分类
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#定义损失含税和优化的方法
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#正式用构建的模型进行分类
tf.global_variables_initializer().run()
for i in range(1000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:#每训练100步进行一次输出，feed_dict就是对placeholder进行赋值
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#在最终测试集上进行训练
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


        





