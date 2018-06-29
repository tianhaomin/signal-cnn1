# -*- coding: utf-8 -*-
"""
Created on Fri May 25 10:11:42 2018

@author: Administrator
"""

import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import tensorflow as tf
train = np.load("F://tmp//data//raw data//train.npy")
label1=np.load("F://tmp//data//raw data//label_train.npy")
test=np.load("F://tmp//data//raw data//test.npy")
label1_test=np.load("F://tmp//data//raw data//label_test.npy")

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

tf.global_variables_initializer().run()



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



###反卷积





def tf_conv2d_transpose(input,weights):
    #input_shape=[n,height,width,channel]
    input_shape = input.shape
    #weights shape=[height,width,out_c,in_c]
    weights_shape=weights.shape
    output_shape=[input_shape[0], input_shape[1] , input_shape[2], weights_shape[2]]

    print("output_shape:",output_shape)

    deconv=tf.nn.conv2d_transpose(input,weights,output_shape=output_shape,
        strides=[1, 1, 1, 1], padding='SAME')
    return deconv


def unpool(x, size):
    out = tf.concat([x, tf.zeros_like(x)], 2)
#    out = tf.concat([out, tf.zeros_like(out)], 2)

    sh = x.get_shape().as_list()
    if None not in sh[1:]:
        out_size = [1, sh[1] * 1, sh[2] * size, sh[3]]
        return tf.reshape(out, out_size)
    
    shv = tf.shape(x); print (sh); print (shv); print (sh[3])
    ret = tf.reshape(out, tf.stack([-1, shv[1] * size, shv[2] * size, 3]))
    ret.set_shape([None, None, None, 3])
    return ret
#def unpool_with_with_argmax(pooled, ind, ksize=[1, 2, 2, 1]):
#    """
#      To unpool the tensor after  max_pool_with_argmax.
#      Argumnets:
#          pooled:    the max pooled output tensor
#          ind:       argmax indices , the second output of max_pool_with_argmax
#          ksize:     ksize should be the same as what you have used to pool
#      Returns:
#          unpooled:      the tensor after unpooling
#      Some points to keep in mind ::
#          1. In tensorflow the indices in argmax are flattened, so that a maximum value at position [b, y, x, c] becomes flattened index ((b * height + y) * width + x) * channels + c
#          2. Due to point 1, use broadcasting to appropriately place the values at their right locations ! 
#    """
#    # Get the the shape of the tensor in th form of a list
#    input_shape = pooled.get_shape().as_list()
#    # Determine the output shape
#    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
#    # Ceshape into one giant tensor for better workability
#    pooled_ = tf.reshape(pooled, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]])
#    # The indices in argmax are flattened, so that a maximum value at position [b, y, x, c] becomes flattened index ((b * height + y) * width + x) * channels + c
#    # Create a single unit extended cuboid of length bath_size populating it with continous natural number from zero to batch_size
#    batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
#    b = tf.ones_like(ind) * batch_range
#    b_ = tf.reshape(b, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
#    ind_ = tf.reshape(ind, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
#    ind_ = tf.concat([b_, ind_],1)
#    ref = tf.Variable(tf.zeros([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]))
#    # Update the sparse matrix with the pooled values , it is a batch wise operation
#    unpooled_ = tf.scatter_nd_update(ref, ind_, pooled_)
#    # Reshape the vector to get the final result 
#    unpooled = tf.reshape(unpooled_, [output_shape[0], output_shape[1], output_shape[2], output_shape[3]])
#    return unpooled
x_input = test[4].reshape(1,200)
y_input = label1_test[4].reshape(1,4)
# layer1 visualization
input_sig = sess1.run(conv1,feed_dict={x:x_input,y_:y_input,keep_prob:1.0})
w1 = sess1.run(W_conv1)
deconv1 = tf_conv2d_transpose(input_sig,w1)
temp = sess1.run(deconv1)
y_plot = temp[0,0,:,0]
x_plot = np.arange(1,201,1)
plt.plot(x_plot,y_plot)
#plt.ylim(-25,0)
plt.show()
plt.plot(x_plot,test[0])
#plt.ylim(-25,0)
plt.show()
#layer2 visulization
input_sig = sess1.run(conv2,feed_dict={x:x_input,y_:y_input,keep_prob:1.0})
w2 = sess1.run(W_conv2)
b1 = sess1.run(b_conv1)
deconv1 = tf_conv2d_transpose(input_sig,w2)
deconv2 = tf.nn.relu(deconv1+b1)
deconv2 = sess1.run(deconv2)
deconv = tf_conv2d_transpose(deconv2,w1)
temp = sess1.run(deconv)
y_plot = temp[0,0,:,0]
x_plot = np.arange(1,201,1)
plt.plot(x_plot,y_plot)
#plt.ylim(-25,0)
plt.show()
plt.plot(x_plot,test[0])
plt.ylim(-25,0)
plt.show()
#layer3 visulization
input_sig = sess1.run(conv4,feed_dict={x:x_input,y_:y_input,keep_prob:1.0})
w3 = sess1.run(W_conv4)
b2 = sess1.run(b_conv2)
deconv1 = tf_conv2d_transpose(input_sig,w3)
unpool1 = unpool(deconv1,2)
deconv2 = tf.nn.relu(unpool1+b2)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w2)
deconv2 = tf.nn.relu(deconv2+b1)
deconv2 = sess1.run(deconv2)
deconv = tf_conv2d_transpose(deconv2,w1)
temp = sess1.run(deconv)
y_plot = temp[0,0,:,0]
x_plot = np.arange(1,201,1)
plt.plot(x_plot,y_plot)
#plt.ylim(-25,0)
plt.show()
plt.plot(x_plot,test[0])
plt.ylim(-25,0)
plt.show()
#layer4 visualization
input_sig = sess1.run(conv5,feed_dict={x:x_input,y_:y_input,keep_prob:1.0})
w4 = sess1.run(W_conv5)
b3 = sess1.run(b_conv4)
deconv1 = tf_conv2d_transpose(input_sig,w4)
deconv2 = tf.nn.relu(deconv1+b3)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w3)
unpool1 = unpool(deconv2,2)
deconv2 = tf.nn.relu(unpool1+b2)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w2)
deconv2 = tf.nn.relu(deconv2+b1)
deconv2 = sess1.run(deconv2)
deconv = tf_conv2d_transpose(deconv2,w1)
temp = sess1.run(deconv)
y_plot = temp[0,0,:,0]
x_plot = np.arange(1,201,1)
plt.plot(x_plot,y_plot)
#plt.ylim(-25,0)
plt.show()
plt.plot(x_plot,test[0])
plt.ylim(-25,0)
plt.show()
#layer5 visualization
input_sig = sess1.run(conv6,feed_dict={x:x_input,y_:y_input,keep_prob:1.0})
w5 = sess1.run(W_conv6)
b4 = sess1.run(b_conv5)
deconv1 = tf_conv2d_transpose(input_sig,w5)
unpool1 = unpool(deconv1,2)
deconv2 = tf.nn.relu(unpool1+b4)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w4)
deconv2 = tf.nn.relu(deconv2+b3)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w3)
unpool1 = unpool(deconv2,2)
deconv2 = tf.nn.relu(unpool1+b2)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w2)
deconv2 = tf.nn.relu(deconv2+b1)
deconv2 = sess1.run(deconv2)
deconv = tf_conv2d_transpose(deconv2,w1)
temp = sess1.run(deconv)
y_plot = temp[0,0,:,0]
x_plot = np.arange(1,201,1)
plt.plot(x_plot,y_plot)
#plt.ylim(-25,0)
plt.show()
plt.plot(x_plot,test[0])
#plt.ylim(-25,0)
plt.show()
#layer6 visualization
input_sig = sess1.run(conv7,feed_dict={x:x_input,y_:y_input,keep_prob:1.0})
w6 = sess1.run(W_conv7)
b5 = sess1.run(b_conv6)
deconv1 = tf_conv2d_transpose(input_sig,w6)
deconv2 = tf.nn.relu(deconv1+b5)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w5)
unpool1 = unpool(deconv2,2)
deconv2 = tf.nn.relu(unpool1+b4)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w4)
deconv2 = tf.nn.relu(deconv2+b3)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w3)
unpool1 = unpool(deconv2,2)
deconv2 = tf.nn.relu(unpool1+b2)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w2)
deconv2 = tf.nn.relu(deconv2+b1)
deconv2 = sess1.run(deconv2)
deconv = tf_conv2d_transpose(deconv2,w1)
temp = sess1.run(deconv)
y_plot = temp[0,0,:,0]
x_plot = np.arange(1,201,1)
plt.plot(x_plot,y_plot)
#plt.ylim(-25,0)
plt.show()
plt.plot(x_plot,test[0])
#plt.ylim(-25,0)
plt.show()
#layer7 visualization
input_sig = sess1.run(conv9,feed_dict={x:x_input,y_:y_input,keep_prob:1.0})
w7 = sess1.run(W_conv9)
b6 = sess1.run(b_conv7)
deconv1 = tf_conv2d_transpose(input_sig,w7)
unpool1 = unpool(deconv1,2)
deconv2 = tf.nn.relu(unpool1+b6)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w6)
deconv2 = tf.nn.relu(deconv2+b5)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w5)
unpool1 = unpool(deconv2,2)
deconv2 = tf.nn.relu(unpool1+b4)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w4)
deconv2 = tf.nn.relu(deconv2+b3)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w3)
unpool1 = unpool(deconv2,2)
deconv2 = tf.nn.relu(unpool1+b2)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w2)
deconv2 = tf.nn.relu(deconv2+b1)
deconv2 = sess1.run(deconv2)
deconv = tf_conv2d_transpose(deconv2,w1)
temp = sess1.run(deconv)
y_plot = temp[0,0,:,0]
x_plot = np.arange(1,201,1)
plt.plot(x_plot,y_plot)
#plt.ylim(-25,0)
plt.show()
plt.plot(x_plot,test[0])
#plt.ylim(-25,0)
plt.show()
#layer8 visualization
input_sig = sess1.run(conv10,feed_dict={x:x_input,y_:y_input,keep_prob:1.0})
w8 = sess1.run(W_conv10)
b7 = sess1.run(b_conv9)
deconv1 = tf_conv2d_transpose(input_sig,w8)
deconv2 = tf.nn.relu(deconv1+b7)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w7)
unpool1 = unpool(deconv2,2)
deconv2 = tf.nn.relu(unpool1+b6)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w6)
deconv2 = tf.nn.relu(deconv2+b5)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w5)
unpool1 = unpool(deconv2,2)
deconv2 = tf.nn.relu(unpool1+b4)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w4)
deconv2 = tf.nn.relu(deconv2+b3)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w3)
unpool1 = unpool(deconv2,2)
deconv2 = tf.nn.relu(unpool1+b2)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w2)
deconv2 = tf.nn.relu(deconv2+b1)
deconv2 = sess1.run(deconv2)
deconv = tf_conv2d_transpose(deconv2,w1)
temp = sess1.run(deconv)
y_plot = temp[0,0,:,0]
x_plot = np.arange(1,201,1)
plt.plot(x_plot,y_plot)
#plt.ylim(-25,0)
plt.show()
plt.plot(x_plot,test[0])
#plt.ylim(-25,0)
plt.show()
#layer9 visulization
input_sig = sess1.run(conv11,feed_dict={x:x_input,y_:y_input,keep_prob:1.0})
w9 = sess1.run(W_conv11)
b8 = sess1.run(b_conv10)
deconv1 = tf_conv2d_transpose(input_sig,w9)
deconv2 = tf.nn.relu(deconv1+b8)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w8)
deconv2 = tf.nn.relu(deconv2+b7)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w7)
unpool1 = unpool(deconv2,2)
deconv2 = tf.nn.relu(unpool1+b6)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w6)
deconv2 = tf.nn.relu(deconv2+b5)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w5)
unpool1 = unpool(deconv2,2)
deconv2 = tf.nn.relu(unpool1+b4)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w4)
deconv2 = tf.nn.relu(deconv2+b3)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w3)
unpool1 = unpool(deconv2,2)
deconv2 = tf.nn.relu(unpool1+b2)
deconv2 = sess1.run(deconv2)
deconv2 = tf_conv2d_transpose(deconv2,w2)
deconv2 = tf.nn.relu(deconv2+b1)
deconv2 = sess1.run(deconv2)
deconv = tf_conv2d_transpose(deconv2,w1)
temp = sess1.run(deconv)
y_plot = temp[0,0,:,0]
x_plot = np.arange(1,201,1)
plt.plot(x_plot,y_plot)
#plt.ylim(-25,0)
plt.show()
plt.plot(x_plot,test[0])
#plt.ylim(-25,0)
plt.show()



