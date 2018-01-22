# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:03:16 2017

@author: Administrator
"""

# knowledge of evey layer
# 建议与卷积模型一样的自编码器然后输出编码之后的结果。
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import os
from keras import optimizers

# All parameter gradients will be clipped to
# a maximum value of 0.5 and
# a minimum value of -0.5.
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)


train = train.reshape(3400,1,200,1)
test = test.reshape(124,1,200,1)
# 编码器
input_x = Input(shape=(1,200,1))  # tensorflow后端
x = Conv2D(32, (1,1), activation='relu', padding='same')(input_x)
x = Conv2D(48, (1,3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2,2),strides=(2,2), padding='same')(x)
x = Conv2D(64, (1,1), activation='relu', padding='same')(x)
x = Conv2D(80, (1,3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2,2),strides=(2,2), padding='same')(x)
x = Conv2D(128, (1,3), activation='relu', padding='same')(x)
x = Conv2D(156, (1,3), activation='relu', padding='same')(x)
#x = MaxPooling2D(pool_size=(2,2),strides=(2,2), padding='same')(x)
x = Conv2D(200, (1,3), activation='relu', padding='same')(x)
x = Conv2D(256, (1,3), activation='relu', padding='same')(x)
x = Conv2D(300, (1,3), activation='relu', padding='same')(x)

encoded = MaxPooling2D(pool_size=(2,2),strides=(2,2), padding='same')(x)
# 解码器
#x = UpSampling2D((1, 2))(encoded)
x = Conv2D(300, (1,3), activation='relu', padding='same')(x)
x = Conv2D(256, (1,3), activation='relu', padding='same')(x)
x = Conv2D(200, (1,3), activation='relu', padding='same')(x)
#x = UpSampling2D((1,2))(x)
x = Conv2D(156, (1,3), activation='relu', padding='same')(x)
x = Conv2D(128, (1,3), activation='relu', padding='same')(x)
x = UpSampling2D((1,2))(x)
x = Conv2D(80, (1,3), activation='relu', padding='same')(x)
x = Conv2D(64, (1,1), activation='relu', padding='same')(x)
x = UpSampling2D((1,2))(x)
x = Conv2D(48, (1,3), activation='relu', padding='same')(x)
x = Conv2D(32, (1,1), activation='relu', padding='same')(x)
#x = UpSampling2D((1, 2))(x)

decoded = Conv2D(1, (1, 3), activation='relu', padding='same')(x)

autoencoder = Model(input_x, decoded)
encoder = Model(inputs=input_x, outputs=encoded) 
autoencoder.compile(optimizer='Adam', loss='mse')  
autoencoder.fit(train, train, epochs=500, batch_size=56, shuffle=True)  
      

encoded_results = encoder.predict(test)  
fig, axes = plt.subplots(nrows=50, ncols=6, figsize=(12, 12))
for row in range(50):
    for col in range(6):
        x = [j for j in range(25)]
        y = [k for k in encoded_results[120,:,:,(6*row+col)][0]]
        axes[row,col].plot(x,y)
fig.tight_layout()
plt.show()



    
    
    
    
    
    
    
    
    