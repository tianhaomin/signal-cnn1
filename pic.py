# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:05:11 2018

@author: Administrator
"""
###VGG InceptionV ResNet Our acc-step
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set(style="ticks")
sns.set(style="whitegrid", palette="pastel", color_codes=True)
x = [0,100,200,300,400,500,600,700,800,900,1000]
VGG=[0,0.5,0.6,0.7,0.74,0.95,0.99,0.99,0.99,0.99,0.99]
Inc=[0,0.56,0.79,0.95,0.96,0.99,0.99,0.995,0.995,1,1]
Res=[0,0.90,0.93,0.94,0.95,0.97,0.975,0.98,0.985,0.99,1.00]
our=[0,0.56,0.75,0.85,0.956,0.99,0.99,0.99,0.99,1,1]
final = [0,0.99,1,1,1,1,1,1,1,1,1]
plt.plot(x,VGG,label="VGG Net(model2)")
plt.plot(x,Inc,label="Inception Net(model3)")
plt.plot(x,Res,label="ResNet(model4)")
plt.plot(x,our,label="our(model1)")
plt.plot(x,final,label="finalResNet")
plt.legend()
plt.show()
###模型效率对比图
Res_time=0.4738
VGG_time=0.88
Inception_time=5.43
our = 0.29
data=[0.4738,0.88,5.43,0.29,0.56]
time=np.arange(5)
fig, ax = plt.subplots()
plt.bar(time, data)
plt.xticks(time, ('ResNet(model4)', 'VGG(model2)', 'Inception(model3)', 'Our(model1)','FinalRes(model5)'))
plt.ylabel("s/50 samplse")
plt.legend()
plt.show()
#模型在第一个场景下的对比图
step=[0,100,300,500,800,1000,1500,2500]
VGG = [0.2,0.4,0.68,0.71,0.755,0.88,0.86,0.92]
Inception=[0.2,0.4,0.46,0.70,0.85,0.86,0.87,0.86]
res=[0.2,0.75,0.88,0.89,0.90,0.89,0.89,0.88]
our=[0.2,0.39,0.77,0.79,0.806,0.85,0.89]
#引入新的信号
#train
x = [0,100,300,500,1000,1500,2000,2500,3000]
vgg = [0.175,0.72,0.925,0.92,0.955,0.97,0.98,0.98,0.985]
inception=[0.19,0.76,0.87,0.86,0.85,0.85,0.89,0.93,0.94,]
res=[0.2,0.87,0.86,0.9,0.943,0.97,0.98,0.972,0.99]
our=[0.23,0.52,0.83,0.84,0.915,0.89,0.91,0.93,0.96]
fianl=[0.43,0.855,0.89,0.88,0.935,0.92,0.915,0.91,0.95]
plt.plot(x,vgg,label="vgg(model2)")
plt.plot(x,inception,label="inception(model3)")
plt.plot(x,res,label="res(model4)")
plt.plot(x,our,label="our(model1)")
plt.plot(x,fianl,label='model5')
plt.legend()
plt.show()
# validation
vgg=[0.2,0.69,0.832,0.85,0.91,0.91,0.923,0.90,0.94,]
inception=[0.2,0.70,0.83,0.845,0.91,0.91,0.923,0.93,0.941]
res=[0.2,0.8,0.85,0.74,0.845,0.8193,0.883,0.875,0.89]
our=[0.2,0.69,0.83,0.85,0.909,0.91,0.923,0.903,0.941]
final=[0.2774,0.87,0.884,0.871,0.871,0.90,0.864,0.884,0.891]
plt.plot(x,vgg,label="vgg(model2)")
plt.plot(x,inception,label="inception(model3)")
plt.plot(x,res,label="res(model4)")
plt.plot(x,our,label="our(model1)")
plt.plot(x,final,label="model5")
plt.legend()
plt.show()

####kernel size comparision
step=[0,50,100,200,300,400,500,600]
#kernel size 1x1 1x3
train1=[0.23,0.73,0.86,0.96,0.99,0.98,0.995,1]
valid1=[0.25,0.92,0.952,0.99,0.992,1,1,1]
#kernel size 1x1 1x7
train2=[0.255,0.675,0.68,0.75,0.87,0.96,0.995,1]
valid2=[0.242,0.63,0.742,0.7334,0.98,0.976,1,1]
#kernel size 1x3,1x7
train3=[0.315,0.45,0.58,0.73,0.72,0.78,0.78,0.80]
valid3=[0.25,0.5,0.49,0.73,0.75,0.75,0.75,0.75]
#kernel size 1x1 1x5
train4=[0.2,0.67,0.87,0.95,0.975,0.985,0.995,0.995]
valid4=[0.25,0.895,0.912,0.98,1,0.984,1,1]
plt.plot(step,train1,label="1x1 1x3")
plt.plot(step,train4,label="1x1 1x5")
plt.plot(step,train2,label="1x1 1x7")
plt.plot(step,train3,label="1x3 1x7")
plt.legend()
plt.show()

plt.plot(step,valid1,label="1x1 1x3")
plt.plot(step,valid4,label="1x1 1x5")
plt.plot(step,valid2,label="1x1 1x7")
plt.plot(step,valid3,label="1x3 1x7")
plt.legend()
plt.show()





