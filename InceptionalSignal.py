# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 18:09:26 2017

@author: Administrator
"""

import tensorflow as tf
# slim可以简洁快速的定义模型
slim = tf.contrib.slim
trunc_normal = lambda stddev:tf.truncated_normal_initializer(0.0,stddev)
# Inception V3 网络结构的卷积部分的构建
def inception_v3_base(inputs,scope=None):
    end_points = { }
    with tf.variable_scope(scope,'InceptionV3',[inputs]):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='VALID'):
            net = slim.conv2d(inputs,32,[1,3],stride=2,scope='conv2d_1a_3x3')
            net = slim.conv2d(net,32,[1,3],scope='Conv2d_2a_3x3')  # 步长默认为1
            net = slim.conv2d(net,64,[1,3],padding='SAME',scope='Conv2d_2b_3x3')  # padding默认是VALID
            net = slim.max_pool2d(net,[1,3],stride=2,scope='MaxPool_3a_3x3')
            net = slim.conv2d(net,80,[1,1],scope='Conv2d_3b_1x1')
            net = slim.conv2d(net,192,[1,3],scope='Conv2d_4a_3x3')
            net = slim.max_pool2d(net,[1,3],stride=2,scope='MaxPool_5a_3x3')
        # 接下来定义INception模块
        with tf.slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='SAME'):
            # 先设置默认参数
            # 第一组 Inception
            # 第一个Inception模块
            with tf.Variable_scope('Mixed_5b'):
                with tf.Variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                with tf.Variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net,48,[1,1],scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1,64,[5,5],scope='Conv2d_0b_5x5')
                with tf.Variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2,96,[1,3],scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2,96,[1,3],scope='Conv2d_0c_3x3')
                with tf.Variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net,[1,3],scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3,32,[1,1],scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
            # 第二个Inception模块
            with tf.Variable_scope('Mixed_5c'):
                with tf.Variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                with tf.Variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net,48,[1,1],scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1,64,[1,5],scope='Conv2d_0b_5x5')
                with tf.Variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2,96,[1,3],scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2,96,[1,3],scope='Conv2d_0c_3x3')
                with tf.Variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net,[1,3],scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3,32,[1,1],scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
            # 第三个Inception模块
            with tf.Variable_scope('Mixed_5d'):
                with tf.Variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                with tf.Variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net,48,[1,1],scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1,64,[1,5],scope='Conv2d_0b_5x5')
                with tf.Variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2,96,[1,3],scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2,96,[1,3],scope='Conv2d_0c_3x3')
                with tf.Variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net,[1,3],scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3,32,[1,1],scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
            # 第二组Inception
            # 第一个Inception model
            with tf.Variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [1, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [1, 3], scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, 96, [1, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [1, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            # 第二个Inception model
            with tf.Variable_scope('Mixed_6b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [1, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [1, 3], scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, 96, [1, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [1, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            # 第三个Inception model
            with tf.Variable_scope('Mixed_6c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [1, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [1, 3], scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, 96, [1, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [1, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            # 第四个Inception model
            with tf.Variable_scope('Mixed_6d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [1, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [1, 3], scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, 96, [1, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [1, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            # 第五个Inception model
            with tf.Variable_scope('Mixed_6e'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [1, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [1, 3], scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, 96, [1, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [1, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            # 第三组Inception
            # 第一个Inception model
            with tf.Variable_scope('Mixed_7a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 320, [1, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 1], scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            # 第二个Inception model
            with tf.Variable_scope('Mixed_7b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 320, [1, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 1], scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [1, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            # 第三个Inception model
            with tf.Variable_scope('Mixed_7c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 320, [1, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 1], scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [1, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            return net,end_points
# 接下来定义平均池化层，全连接层
def inception_v3(inputs,
                 num_classes=7,#需要区分的种类
                 is_training=True,
                 dropout_keep_prob=0.8,#需要保留的节点数默认保留80%的节点
                 prediction_fn=slim.softmax,#多分类的函数，默认softmax
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV3'):

  with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_v3_base(inputs, scope=scope)

      # Auxiliary Head logits
      #辅助节点分类部分对分类结果预测有很大帮助
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                          stride=1, padding='SAME'):
        aux_logits = end_points['Mixed_6e']
        with tf.variable_scope('AuxLogits'):
          aux_logits = slim.avg_pool2d(
              aux_logits, [5, 5], stride=3, padding='VALID',
              scope='AvgPool_1a_5x5')
          aux_logits = slim.conv2d(aux_logits, 128, [1, 1],
                                   scope='Conv2d_1b_1x1')

          # Shape of feature map before the final layer.
          aux_logits = slim.conv2d(
              aux_logits, 768, [5,5],
              weights_initializer=trunc_normal(0.01),
              padding='VALID', scope='Conv2d_2a_5x5')
          aux_logits = slim.conv2d(
              aux_logits, num_classes, [1, 1], activation_fn=None,
              normalizer_fn=None, weights_initializer=trunc_normal(0.001),
              scope='Conv2d_2b_1x1')
          if spatial_squeeze:
            aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
          end_points['AuxLogits'] = aux_logits

      # Final pooling and prediction
      #正常的分类预测逻辑
      with tf.variable_scope('Logits'):
        net = slim.avg_pool2d(net, [8, 8], padding='VALID',
                              scope='AvgPool_1a_8x8')
        # 1 x 1 x 2048
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        end_points['PreLogits'] = net
        # 2048
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1')
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        # 1000
      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points


inception_v3(train)

