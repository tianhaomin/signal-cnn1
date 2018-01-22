# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:12:08 2017

@author: Administrator
"""
# 用残差网络效果很差

import collections
import tensorflow as tf
slim = tf.contrib.slim

import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn
import os
from pandas import DataFrame
import tensorflow as tf
import random
from sklearn import preprocessing 
#a = os.listdir("F:/project/Yin/spectrum-data")
#def read_data(file_num,start_frq,end_frq):
#    z1 = DataFrame({})
#    for i in range(file_num):
#        df = pandas.read_table("F:/project/Yin/spectrum-data//"+a[i],names=["fc","E"])
#        df1 = df[df.fc.between(start_frq,end_frq)]
#        z1 = pandas.concat([z1,df1['E']])
#        z2 = z1.values
#        z3 = np.array([z2])
#    return z3
#############cdma_down#########2
#data_cdma_down = read_data(850,870,880).reshape(850,401)
##############egsm_down#########4
#data_egsm_down = read_data(850,930,940).reshape(850,401)
#data_wcdma_down = read_data(850,2135,2145).reshape(850,401)
###############4G################6
#data_lte_down = read_data(850,1850,1860).reshape(850,401)
##data_evdo_down = read_data(850,1920,1930).reshape(850,401)
#data_dcs_down = read_data(850,1900,1910).reshape(850,401)
#
#data_cdma_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_cdma_down] )
#data_egsm_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_egsm_down] )
#data_wcdma_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_wcdma_down] )
#data_lte_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_lte_down] )
##data_evdo_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_evdo_down] )
#data_dcs_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in data_dcs_down] )
#
#train_data = [data_cdma_down,data_egsm_down,data_wcdma_down,data_lte_down,data_dcs_down]
#train = train_data[0]
#for i in range(len(train_data)-1):
#    train = np.vstack((train,train_data[i+1]))
########test set#############
#a = os.listdir("F:/project/Yin/spectrum-data")
#def read_data(file_num,start_frq,end_frq):
#    z1 = DataFrame({})
#    for i in range(file_num):
#        df = pandas.read_table("F:/project/Yin/spectrum-data//"+a[i+850],names=["fc","E"])
#        df1 = df[df.fc.between(start_frq,end_frq)]
#        z1 = pandas.concat([z1,df1['E']])
#        z2 = z1.values
#        z3 = np.array([z2])
#    return z3
#
#############cdma_down#########2
#test_cdma_down = read_data(31,870,880).reshape(31,401)
##############egsm_down#########4
#test_egsm_down = read_data(31,930,940).reshape(31,401)
###########lte下行#########
#test_lte_down = read_data(31,1850,1860).reshape(31,401)
##########wcdma下行###########
#test_wcdma_down = read_data(31,2135,2145).reshape(31,401)
##test_evdo_down = read_data(31,1920,1930).reshape(31,401)
#test_dcs_down = read_data(31,1900,1910).reshape(31,401)
#
########
#test_cdma_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_cdma_down] )
#test_egsm_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_egsm_down] )
#test_lte_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_lte_down] )
#test_wcdma_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_wcdma_down] )
##test_evdo_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_evdo_down] )
#test_dcs_down = np.array( [[row[i] for i in range(0, 201) if i != 200] for row in test_dcs_down] )
###############
#test_data = [test_cdma_down,test_egsm_down,test_wcdma_down,test_lte_down,test_dcs_down]
#
#test = test_data[0]
#for i in range(len(test_data)-1):
#    test = np.vstack((test,test_data[i+1]))
########################################shuffle#######
#train=train.astype(np.float32)
#test = test.astype(np.float32)
#enc = preprocessing.OneHotEncoder()  
#enc.fit([[1],[2],[3],[4],[5]])  
#
#array1_1 = enc.transform([[1]]*850).toarray()  
#array1_2 = enc.transform([[2]]*850).toarray()  
#array1_3 = enc.transform([[3]]*850).toarray()  
#array1_4 = enc.transform([[4]]*850).toarray()  
#array1_5 = enc.transform([[5]]*850).toarray()  
##array1_6 = enc.transform([[6]]*850).toarray()  
#label1 = np.vstack((array1_1,array1_2,array1_3,array1_4,array1_5))
#array1_1_test = enc.transform([[1]]*31).toarray()
#array1_2_test = enc.transform([[2]]*31).toarray()
#array1_3_test = enc.transform([[3]]*31).toarray()
#array1_4_test = enc.transform([[4]]*31).toarray()
#array1_5_test = enc.transform([[5]]*31).toarray()
##array1_6_test = enc.transform([[6]]*31).toarray()
#label1_test = np.vstack((array1_1_test,array1_2_test,array1_3_test,array1_4_test,array1_5_test))
## 顺序打乱
#label1 = label1.astype(np.float32)
#label1_test = label1_test.astype(np.float32)
#
#
#idx=random.sample(range(4250),4250) 
#idx1 = random.sample(range(155),155)
#train = train[idx]
#label1 = label1[idx]
#test = test[idx1]
#label1_test = label1_test[idx1]
#block类定义的是一个残差学习单元。scope是这个单元的名字，unit_fn就是这个单元的参数设置
#每个残差网络单元有三个卷积层，所以[(256,64,1)]x2+[(256,64,2)]代表有三个残差单元
#（256,64,1）代表这个残差单元的三个卷积层中第一层和第二层卷积核数是64，第三层卷积核数是
#256，第二层步长为1
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """A named tuple describing a ResNet block.

  Its parts are:
    scope: The scope of the `Block`.
    unit_fn: The ResNet unit function which takes as input a `Tensor` and
      returns another `Tensor` with the output of the ResNet unit.
    args: A list of length equal to the number of units in the `Block`. The list
      contains one (depth, depth_bottleneck, stride) tuple for each unit in the
      block to serve as argument to unit_fn.
  """


def subsample(inputs, factor, scope=None):
  """Subsamples the input along the spatial dimensions.

  Args:
    inputs: A `Tensor` of size [batch, height_in, width_in, channels].
    factor: The subsampling factor.
    scope: Optional variable_scope.

  Returns:
    output: A `Tensor` of size [batch, height_out, width_out, channels] with the
      input, either intact (if factor == 1) or subsampled (if factor > 1).
  """
  if factor == 1:
    return inputs
  else:
    return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
  """Strided 2-D convolution with 'SAME' padding.

  When stride > 1, then we do explicit zero-padding, followed by conv2d with
  'VALID' padding.

  Note that

     net = conv2d_same(inputs, num_outputs, 3, stride=stride)

  is equivalent to

     net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
     net = subsample(net, factor=stride)

  whereas

     net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')

  is different when the input's height or width is even, which is why we add the
  current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters.
    kernel_size: An int with the kernel_size of the filters.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    scope: Scope.

  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """
  if stride == 1:
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                       padding='SAME', scope=scope)
  else:
    #kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs,
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                       padding='VALID', scope=scope)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks,
                       outputs_collections=None):
  """Stacks ResNet `Blocks` and controls output feature density.

  First, this function creates scopes for the ResNet in the form of
  'block_name/unit_1', 'block_name/unit_2', etc.


  Args:
    net: A `Tensor` of size [batch, height, width, channels].
    blocks: A list of length equal to the number of ResNet `Blocks`. Each
      element is a ResNet `Block` object describing the units in the `Block`.
    outputs_collections: Collection to add the ResNet block outputs.

  Returns:
    net: Output tensor 

  """
  for block in blocks:
    with tf.variable_scope(block.scope, 'block', [net]) as sc:
      for i, unit in enumerate(block.args):

        with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
          unit_depth, unit_depth_bottleneck, unit_stride = unit
          net = block.unit_fn(net,
                              depth=unit_depth,
                              depth_bottleneck=unit_depth_bottleneck,
                              stride=unit_stride)
      net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
      
  return net


def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  """Defines the default ResNet arg scope.

  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.

  Args:
    is_training: Whether or not we are training the parameters in the batch
      normalization layers of the model.
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.

  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'is_training': is_training,
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # slim.arg_scope([slim.max_pool2d], padding='VALID').
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc




@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collections=None, scope=None):
  """Bottleneck residual unit variant with BN before convolutions.

  This is the full preactivation residual unit variant proposed in [2]. See
  Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
  variant which has an extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
    if depth == depth_in:
      shortcut = subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(preact, depth, 1, stride=stride,
                             normalizer_fn=None, activation_fn=None,
                             scope='shortcut')

    residual = slim.conv2d(preact, depth_bottleneck, 1, stride=1,
                           scope='conv1')
    residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                                        scope='conv2')
    residual = slim.conv2d(residual, depth, 1, stride=1,
                           normalizer_fn=None, activation_fn=None,
                           scope='conv3')

    output = shortcut + residual

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


def resnet_v2(inputs,
              blocks,
              num_classes=None,
              global_pool=True,
              include_root_block=True,
              reuse=None,
              scope=None):
  """Generator for v2 (preactivation) ResNet models.

  This function generates a family of ResNet v2 models. See the resnet_v2_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.


  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it. If excluded, `inputs` should be the
      results of an activation-less convolution.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.


  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         stack_blocks_dense],
                        outputs_collections=end_points_collection):
      net = inputs
      if include_root_block:
        # We do not include batch normalization or activation functions in conv1
        # because the first ResNet unit will perform these. Cf. Appendix of [2].
        with slim.arg_scope([slim.conv2d],
                            activation_fn=None, normalizer_fn=None):
          net = conv2d_same(net, 64, 1, stride=2, scope='conv1')
        net = slim.max_pool2d(net, [1, 3], stride=2, scope='pool1')
      net = stack_blocks_dense(net, blocks)
      # This is needed because the pre-activation variant does not have batch
      # normalization or activation functions in the residual unit output. See
      # Appendix of [2].
      net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
      if global_pool:
        # Global average pooling.
        net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
      if num_classes is not None:
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                          normalizer_fn=None, scope='logits')
      # Convert end_points_collection into a dictionary of end_points.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if num_classes is not None:
        end_points['predictions'] = slim.softmax(net, scope='predictions')
      return net, end_points



def resnet_v2_50(inputs,
                 num_classes=None,
                 global_pool=False,
                 reuse=None,
                 scope='resnet_v2_50'):
  """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      Block('block1', bottleneck, [(64, 32, 1)] * 3 + [(64, 32, 2)]),
      Block(
          'block2', bottleneck, [(128, 64, 1)] * 1 + [(128, 64, 2)]),
      Block(
          'block3', bottleneck, [(256, 128, 1)] * 1 + [(256, 128, 2)]),
      Block(
          'block4', bottleneck, [(512, 256, 1)] * 1)]
  return resnet_v2(inputs, blocks, num_classes, global_pool,
                   include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_101(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_101'):
  """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      Block(
          'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
      Block(
          'block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
      Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)]
  return resnet_v2(inputs, blocks, num_classes, global_pool,
                   include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_152(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_152'):
  """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      Block(
          'block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
      Block(
          'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
      Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)]
  return resnet_v2(inputs, blocks, num_classes, global_pool,
                   include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_200(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_200'):
  """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
  blocks = [
      Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      Block(
          'block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
      Block(
          'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
      Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)]
  return resnet_v2(inputs, blocks, num_classes, global_pool,
                   include_root_block=True, reuse=reuse, scope=scope)

  


sess1  = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 200],name='x-input')
y_ = tf.placeholder(tf.float32, [None, 5],name='y-input')
is_train = tf.placeholder(tf.bool,name='is_train')
x_image = tf.reshape(x,[-1,1,200,1])
with slim.arg_scope(resnet_arg_scope(is_training=is_train)):
    net, end_points = resnet_v2_50(x_image,5)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(end_points['predictions'][:,0,0,:],1e-10,1.0)), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(end_points['predictions'][:,0,0,:], 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




tf.global_variables_initializer().run()
#import time

result_test=[]
result_train=[]
for i in range(3000):
    for j in range(85):
        if j % 10 == 0:  # Record summaries and test-set accuracy
            acc = sess1.run([accuracy], feed_dict={x:test,y_:label1_test,is_train:True})
            result_test.append(acc)
            print('Accuracy at step %s of epoch %s: %s' % (j,i, acc))
        else:  # Record a summary
#            t1=time.time()
            _ = sess1.run([train_step], feed_dict={x:train[50*j:50*(j+1)],y_:label1[50*j:50*(j+1)],is_train:True})
            acc_train = sess1.run([accuracy],feed_dict={x:train[0:200],y_:label1[0:200],is_train:True})
            result_train.append(acc_train)
            print(acc_train)
                #print(2)
#            t2=time.time()
#            print(t2-t1)

sess1.close()

 
#batch_size = 32
#height, width = 224, 224
#inputs = tf.random_uniform((batch_size, height, width, 3))
#with slim.arg_scope(resnet_arg_scope(is_training=False)):
#   net, end_points = resnet_v2_152(inputs, 1000)
#  
#init = tf.global_variables_initializer()
#sess = tf.Session()
#sess.run(init)  
#num_batches=100
#time_tensorflow_run(sess, net, "Forward") 
#import numpy as np
#train = np.load("F://tmp//data//new data//train.npy")
#label1=np.load("F://tmp//data//new data//label_train.npy")
#test=np.load("F://tmp//data//new data//test.npy")
#label1_test=np.load("F://tmp//data//new data//label_test.npy")
#
#np.save("F://tmp//data//add new signal//res_final_5train.npy",result_train)
#np.save("F://tmp//data//add new signal//res_final_5test.npy",result_test)
#np.save("F://tmp//data//new data//train.npy",train)
#np.save("F://tmp//data//new data//label_train.npy",label1)
#np.save("F://tmp//data//new data//test.npy",test)
#np.save("F://tmp//data//new data//label_test.npy",label1_test)