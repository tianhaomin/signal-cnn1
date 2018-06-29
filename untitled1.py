# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:14:24 2018

@author: Administrator
"""
import matplotlib.pyplot as plt
import pandas 
import seaborn as sns
#ns.set_style("whitegrid")
sns.set()  
df = pandas.read_table("F://project//Yin//spectrum-data//20160921_165221_117.7044_38.9908_1.txt",names=['fc','E'])
x = df['fc'].values
y = df['E'].values
df1 = df[df.fc.between(100,110)]
x2 = df1['fc'].values
y2 = df1['E'].values
plt.plot(x2,y2)
plt.show()  



import math
from matplotlib import pyplot as plt  
import numpy as np  
from mpl_toolkits.mplot3d import axes3d

low=lambda x:10000 if x>10000 else -10000 if x<-10000 else x

f=lambda x:(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))#设计一个函数

start=-2 #输入需要绘制的起始值（从左到右）
stop=2 #输入需要绘制的终点值
step=0.01#输入步长


num=(stop-start)/step #计算点的个数
x = np.linspace(start,stop,num)
y = f(x)

#for i in range(len(y)):#再应用一个low函数以防止函数值过大导致显示错误（可选）#若函数无法直接应用到np数组上可以使用for来逐个应用
#    y[i]=low(y[i])
#z=y

fig=plt.figure(figsize=(6,6))#建立一个对象并设置窗体的大小，使其为正方形，好看 #注意 可以建立多个对象，但plt指令只会对最后一个指定的对象进行操作（查看过源码了）
plt.grid(True)#显示网格
plt.axis("equal")#设置了x、y刻度长度一致#需要放在x、ylim指令前
plt.xlim((-3, 3))#显示的x的范围（不设置则由程序自动设置）
plt.ylim((-1, 2))#显示的y的范围
plt.plot(x, y,label='First Curve')#在当前的对象上进行操作

plt.savefig("F:\project\Cnn\pic\d13.jpg")
#plt.plot([2*min(x),2*max(x)], [0,0],label='x-axis')#用定义域最长距离的两倍作出x轴
#plt.plot([0,0],[2*min(y),2*max(y)],label='y-axis')#用值域最长距离的两倍作出y轴
plt.legend()#显示旁注#注意：不会显示后来再定义的旁注
plt.show(fig)#没有输入值默认展示所有对象 #注意：plt.show()之后再次使用plt.show()指令将不会展示任何对象，若想再次展示对象，可以对对象使用fig.show()


x = np.arange(-10,10,1)
y = []
for i in x:
    if i<=0:
        y.append(0)
    else:
        y.append(1)
        
plt.plot(x,y)        


