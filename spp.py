# -*- coding: utf-8 -*-
"""
Created on Thu May 17 15:14:37 2018

@author: Administrator
"""

###SPP layer
from __future__ import absolute_import  
import math                                                                       
from __future__ import division
from __future__ import print_function

import os
import logging
import math 
import sys

import numpy as np
import tensorflow as tf

class SPPLayer():
    def __init__(self,bins,feature_map_size):
        self.strides = []
        self.filters = []
#        print(type(feature_map_size))
        self.a = float(feature_map_size)
        self.bins = bins
        self.n = len(bins)

    def spatial_pyramid_pooling(self,data):
        self.input = data
        self.batch_size = self.input.get_shape().as_list()[0]
        for i in range(self.n):
            x = int(math.floor(self.a/float(self.bins[i])))
            self.strides.append(x)
            x = int (math.ceil(self.a/float(self.bins[i])))
            self.filters.append(x)

        self.pooled_out = []
        for i in range(self.n):
            self.pooled_out.append(tf.nn.max_pool(self.input,
                ksize=[1, self.filters[i], self.filters[i], 1], 
                strides=[1, self.strides[i], self.strides[i], 1],
                padding='VALID'))

        for i in range(self.n):
            self.pooled_out[i] = tf.reshape(self.pooled_out[i], [self.batch_size, -1])
        
        self.output = tf.concat(1, [self.pooled_out[0], self.pooled_out[1], self.pooled_out[2]])

        return self.output

#SPP-Net
SEED = 1356
VGG_MEAN = [103.939, 116.779, 123.68]
stddev = 0.05
class SPPnet:
    def __init__(self,model_file=None):
        self.random_weight= False
        if model_file is None:
            self.random_weight = True
            logging.error('please input model file')
        if not os.path.isfile(model_file):
            logging.error(('model file is not exist:'), model_file)
        self.wd = 5e-4
        self.stddev = 0.05
        self.param_dict = np.load(model_file).item()
        print('model file loaded')
    
    def _conv_layer(self, bottom, name, shape=None):
        with tf.variable_scope(name) :
            if shape ==None :
                filter1 = self.get_conv_filter(name)
                conv_bias = self.get_bias(name)
            else :
                initW = tf.truncated_normal_initializer(stddev = self.stddev)
                filter1 = tf.get_variable(name='filter', shape=shape, initializer=initW)
                
                initB = tf.constant_initializer(0.0)
                conv_bias = tf.get_variable(name='bias',shape=shape[3], initializer=initB)
            conv = tf.nn.conv2d(bottom, filter1, strides=[1 ,1 ,1 ,1], padding='SAME')
            relu = tf.nn.relu( tf.nn.bias_add(conv, conv_bias) )
            
            return relu
    def _fc_layer(self, bottom, name, shape=None):
        with tf.variable_scope(name) :
            if shape == None:
                weight = self.get_fc_weight(name)
                bias = self.get_bias(name)
            else:
                weight =self._variable_with_weight_decay(shape, self.stddev, self.wd)
                initB = tf.constant_initializer(0.0)
                bias = tf.get_variable(name='bias',shape=shape[1], initializer=initB)

            fc = tf.nn.bias_add(tf.matmul(bottom, weight), bias)
            
            if name == 'fc8' :
                return fc
            else:
                relu = tf.nn.relu(fc)
                return relu

    def inference(self, data, train=True, num_class=4):
        with tf.name_scope('Processing'):
            self.conv1_1 = self._conv_layer(data, "conv1_1", [1,1,1,32])
            self.conv1_2 = self._conv_layer(self.conv1_1, 'conv1_2', [1,3,32,48])
            self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1,2,2,1],strides=[1,2,2,1],
                    padding='SAME',name='pool1')

            self.conv2_1 = self._conv_layer(self.pool1, 'conv2_1', [1,1,48,64])
            self.conv2_2 = self._conv_layer(self.conv2_1, 'conv2_2', [1,3,64,80])
            self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1,2,2,1],strides=[1,2,2,1],
                    padding='SAME',name='pool2')

            self.conv3_1 = self._conv_layer(self.pool2, 'conv3_1', [1,1,80,128])
            self.conv3_2 = self._conv_layer(self.conv3_1, 'conv3_2', [1,3,128,156])
            self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1,2,2,1],strides=[1,2,2,1],
                    padding='SAME',name='pool3')

            self.conv4_1 = self._conv_layer(self.pool3, 'conv4_1', [1,3,156,200])
            self.conv4_2 = self._conv_layer(self.conv4_1, 'conv4_2', [1,3,200, 256])
            self.conv4_3 = self._conv_layer(self.conv4_2, 'conv4_3', [1,3,256,300])
                        
            bins = [3, 2, 1]
            map_size = self.conv4_3.get_shape().as_list()[2]
            print(self.conv4_3.get_shape())
            sppLayer = SPPLayer(bins, map_size)
            self.sppool = sppLayer.spatial_pyramid_pooling(self.conv4_3)
            
            numH = self.sppool.get_shape().as_list()[1]
            print(numH)
            self.fc5 = self._fc_layer(self.sppool, 'fc5', shape=[numH, 4096])
            if train:
                self.fc5 = tf.nn.dropout(self.fc5, 0.5, seed=SEED)
            
            self.fc6 = self._fc_layer(self.fc5, 'fc6',shape= [4096,4096])
            if train:
                self.fc6 = tf.nn.dropout(self.fc6, 0.5, seed=SEED)
            self.fc7 = self._fc_layer(self.fc6, 'fc7', shape=[4096,num_class])
            print('inference')
            return self.fc7
    
    def loss(self, logits, label=None):
            self.pred = tf.nn.softmax(logits)
            if label is not None:
                label = tf.cast(label, tf.int64)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits, label, name = 'cross_entropy_all')
                self.entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
#                tf.add_to_collection('losses', self.entropy_loss)
#                self.all_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
                
                correct_prediction = tf.equal(tf.argmax(logits,1), label)
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                return (self.entropy_loss, self.accuracy)
            else:
                return self.pred
    
    def train(self, loss, global_step):
        self.lr = tf.train.exponential_decay(self.lr, 
                global_step*self.batch_size, self.train_size*self.decay_epochs, 0.95, staircase=True)
        self.optimizer = tf.train.MomentumOptimizer(self.lr, 0.9).minimize(loss,
                global_step = global_step)
        return (self.optimizer, self.lr)

    def set_lr(self, lr, batch_size, train_size, decay_epochs = 10):
        self.lr = lr
        self.batch_size = batch_size
        self.train_size = train_size
        self.decay_epochs = decay_epochs
    def get_conv_filter(self, name):
        init = tf.constant_initializer(value=self.param_dict[name][0], dtype=tf.float32)
        shape = self.param_dict[name][0].shape
        print('conv Layer name: %s' % name)
        print('conv Layer shape: %s' % str(shape))
        var = tf.get_variable(name = 'filter', initializer=init, shape=shape)
#        if not tf.get_variable_scope().reuse:
#            weight_decay = tf.mul(tf.nn.l2_loss(var), self.wd,
#                                  name='weight_loss')
#            tf.add_to_collection('losses', weight_decay)

        return var
    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.param_dict[name][0], dtype=tf.float32)
        shape = self.param_dict[name][0].shape
        print('fc Layer name: %s' % name)
        print('fc Layer shape: %s' % str(shape))
        var = tf.get_variable(name = 'weight', initializer=init, shape=shape)
#        if not tf.get_variable_scope().reuse:
#            weight_decay = tf.mul(tf.nn.l2_loss(var), self.wd,
#                                  name='weight_loss')
#            tf.add_to_collection('losses', weight_decay)

        return var
    def get_bias(self,name):
        init = tf.constant_initializer(value=self.param_dict[name][1], dtype=tf.float32)
        shape = self.param_dict[name][1].shape
        var = tf.get_variable(name = 'bias', initializer=init, shape=shape)
        return var

    def _variable_with_weight_decay(self, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """

        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape,
                              initializer=initializer)

#        if wd and (not tf.get_variable_scope().reuse):
#            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
#            tf.add_to_collection('losses', weight_decay)
        return var

#train model
train_size = 3060
batch_size = 4
max_epochs =2
num_class = 4
eval_frequency = 100
max_steps = 10000
import time 


def train(train1,label1):
    global_step = tf.Variable(0, trainable=False)
    spp_net = SPPnet
    spp_net.set_lr(spp_net,0.0001, 4, 3000)
# load data
    print('load data')
    train_data = train1
    train_label = label1
    print("load done")
#    valid_data, valid_label, vshape = input_data_t('data/101_ObjectCategories/','data/valid.txt', 32)
    num_class = 4

# train
    print('train')
    logits = spp_net.inference(spp_net,train_data, True, num_class)
    loss, accuracy = spp_net.loss(spp_net,logits, train_label)
    opt, lr = spp_net.train(spp_net,loss, global_step)
    print('train done')
# evaluation
#    eval_logits = spp_net.inference(valid_data, False, num_class)
#    eval_accuracy = spp_net.loss(eval_logits, valid_label)
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        init = tf.initialize_all_variables()
        coord = tf.train.Coordinator()
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        
        start_time = time.time()
    #    print((FLAGS.max_epochs * train_size) // batch_size)
        for step in xrange(max_steps):
            _, loss_value, accu = sess.run([opt, loss, accuracy])
            if step % eval_frequency ==0:
                stop_time = time.time() - start_time
                start_time = time.time()
                print('epoch: %.2f , %.2f ms' % (step * batch_size /train_size,
                    1000 * stop_time / eval_frequency)) 
                print('train loss: %.3f' % loss_value) 
                print('train accu: %.2f%%' % accu)         
        coord.request_stop()
        coord.join(threads)
'''
                if step % eval_frequency == 0:
                stop_time = time.time() - start_time
                start_time = time.time()
                print('epoch: %.2f , %.2f ms' % (step * batch_size /train_size,
                                                1000 * stop_time / eval_frequency)) 
                print('train loss: %.3f' % loss_value) 
                print('train error: %.2f%%' % accu)         
                eval_accu = sess.run([eval_accuracy])
                print('valid error: %.2f%%' % eval_accu)
                
                if((1- eval_accu) < best_error_rate):
                    if((1-eval_accu) < best_error_rate * 0.95):
                        if(patience<step *2):
                            patience = patience *2
                    best_error_rate = 1 - eval_accu 

            if step >= patience:
                saver.save(sess, model_save_file, global_step = step)
                break
            if (step +1) == (FLAGS.max_epochs * train_size) // batch_size:
                saver.save(sess, model_save_file, global_step = step) '''

if __name__ == '__main__':
    #数据读取
    train1 = np.load("F://tmp//data//raw data//train.npy")
    label1=np.load("F://tmp//data//raw data//label_train.npy")
    test=np.load("F://tmp//data//raw data//test.npy")
    label1_test=np.load("F://tmp//data//raw data//label_test.npy")
    train(train1,label1)
