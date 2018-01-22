# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 10:24:43 2017

@author: Administrator
"""
import pandas 
import os
import numpy as np
import seaborn 
import matplotlib.pyplot as plt
a = os.listdir("F:/project/Yin/spectrum-data")
for i in range(100):
    df = pandas.read_table("F:/project/Yin/spectrum-data//"+a[i],names=["fc","E"])
    df1 = df[df.fc.between(1900,1950)]
    plt.plot(df1['fc'],df1['E'])
    plt.show()