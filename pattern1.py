# -*- coding: utf-8 -*-
"""
Created on Thu May  3 09:28:20 2018

@author: Administrator
"""
import numpy as np
import tensorflow as tf
train = np.load("F://tmp//data//raw data//train.npy")
label1=np.load("F://tmp//data//raw data//label_train.npy")
test=np.load("F://tmp//data//raw data//test.npy")
label1_test=np.load("F://tmp//data//raw data//label_test.npy")

learning_rate = 1e-5
inputs_ = tf.placeholder(tf.float32, (None,  1,200,1), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, 1,200,1), name='targets')
### Encoder
conv1 = tf.layers.conv2d(inputs=inputs_, filters=32, kernel_size=(1,1), padding='same', activation=tf.nn.relu)
conv2 = tf.layers.conv2d(inputs=conv1, filters=48, kernel_size=(1,3), padding='same', activation=tf.nn.relu)
# Now 100x100
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
# Now 14x14x32
conv3 = tf.layers.conv2d(inputs=maxpool1, filters=64, kernel_size=(1,1), padding='same', activation=tf.nn.relu)
conv4 = tf.layers.conv2d(inputs=conv3, filters=80, kernel_size=(1,3), padding='same', activation=tf.nn.relu)
# Now 50x50x50
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
conv5 = tf.layers.conv2d(inputs=maxpool2, filters=128, kernel_size=(1,1), padding='same', activation=tf.nn.relu)
conv6 = tf.layers.conv2d(inputs=conv5, filters=156, kernel_size=(1,3), padding='same', activation=tf.nn.relu)
#25x25
maxpool3 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
conv7 = tf.layers.conv2d(inputs=maxpool3, filters=200, kernel_size=(1,3), padding='same', activation=tf.nn.relu)
conv8 = tf.layers.conv2d(inputs=conv7, filters=256, kernel_size=(1,3), padding='same', activation=tf.nn.relu)
conv9 = tf.layers.conv2d(inputs=conv8, filters=300, kernel_size=(1,3), padding='same', activation=tf.nn.relu)
#13x13
encoded = tf.layers.max_pooling2d(conv9, pool_size=(2,2), strides=(2,2), padding='same')

### Decoder
upsample1 = tf.image.resize_images(encoded, size=(1,25), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 7x7x16
conv10 = tf.layers.conv2d(inputs=upsample1, filters=300, kernel_size=(1,3), padding='same', activation=tf.nn.relu)
conv11 = tf.layers.conv2d(inputs=conv10, filters=256, kernel_size=(1,3), padding='same', activation=tf.nn.relu)
conv12 = tf.layers.conv2d(inputs=conv11, filters=200, kernel_size=(1,3), padding='same', activation=tf.nn.relu)
# Now 7x7x16
upsample2 = tf.image.resize_images(conv12, size=(1,50), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 14x14x16
conv13 = tf.layers.conv2d(inputs=upsample2, filters=156, kernel_size=(1,3), padding='same', activation=tf.nn.relu)
conv14 = tf.layers.conv2d(inputs=conv13, filters=128, kernel_size=(1,1), padding='same', activation=tf.nn.relu)
# Now 14x14x32
upsample3 = tf.image.resize_images(conv14, size=(1,100), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 28x28x32
conv15 = tf.layers.conv2d(inputs=upsample3, filters=80, kernel_size=(1,3), padding='same', activation=tf.nn.relu)
conv16 = tf.layers.conv2d(inputs=conv15, filters=64, kernel_size=(1,1), padding='same', activation=tf.nn.relu)
upsample4 = tf.image.resize_images(conv16, size=(1,200), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
conv17 = tf.layers.conv2d(inputs=upsample4, filters=48, kernel_size=(1,3), padding='same', activation=tf.nn.relu)
conv18 = tf.layers.conv2d(inputs=conv17, filters=32, kernel_size=(1,1), padding='same', activation=tf.nn.relu)
# Now 28x28x32
logits = tf.layers.conv2d(inputs=conv18, filters=1, kernel_size=(3,3), padding='same', activation=None)
#Now 28x28x1
# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.relu(logits)
# Pass logits through sigmoid and calculate the cross-entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
epochs = 1000
batch_size = 50
# Set's how much noise we're adding to the MNIST images
noise_factor = 0.5
sess.run(tf.global_variables_initializer())
for e in range(epochs):
    for ii in range(len(train)//batch_size):
        batch = train[ii*50:(ii+1)*50]
        # Get images from the batch
        imgs = batch[0].reshape((-1, 1, 200, 1))
        
        # Add random noise to the input images
        noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
        # Clip the images to be between 0 and 1
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        
        #Noisy images as inputs, original images as targets
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: noisy_imgs,
                                                         targets_: imgs})
    print("Epoch: {}/{}...".format(e+1, epochs),
              "Training loss: {:.4f}".format(batch_cost))