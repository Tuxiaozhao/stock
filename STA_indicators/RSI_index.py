#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: RSI_index.py
@time: 2018/12/07
"""
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
from stock_former.get_mian_indictors import *

'''
假设A为N日内收盘价的正数之和，B为N日内收盘价的负数之和乘以（-1）,
这样，A和B均为正，将A、B代入RSI计算公式，则
RSI（N）=A÷（A＋B）×100
'''

def RSI_N(close, N=5):

    result_rsi = []
    diff = np.diff(close)
    print(diff)
    for i in range(1, len(close)):
        if i < N:
            temp = diff[:i]
            A = sum(np.where(temp>0, temp, 0))
            B = sum(np.where(temp<=0,temp, 0))*(-1)
            result_rsi.append(A/(A+B)*100)
        else:
            temp = diff[i-N:i]
            A = sum(np.where(temp >0, temp, 0))
            B = sum(np.where(temp <= 0, temp, 0))*(-1)
            result_rsi.append(A / (A + B) * 100)
    return  result_rsi

#data = [1,2,1,4,1,6,1,8,100]

info = getdata_by_code('002446',[])

data = info[-61:]
np_data = np.copy(data)
print(np_data[:, 5])
close = np_data[:, 5]
a = RSI_N(np_data[:, 5], 5)
np_a = np.copy(a)/10
print(np_a)
plt.plot(range(len(a)), np_a, label='rsi')
plt.plot(range(len(a)), close[1:], label='close')
plt.legend()
plt.show()
