#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: Mean_line.py
@time: 2018/11/26
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 每天的收盘价均值的变化
# 返回的数据都是对应的一天的数值




def get_maN(close, N=5):
    # 平均权重
    weights = np.ones(N) / N
    sma = np.convolve(weights, close)[N - 1:-N + 1]
    return sma

# ma5 返回的值是今天以及前四天的收盘假的平均值
# day_index 是对应的closes中求取均值的下标
# 短期的移动平均线有： ma3,ma5,ma10, 中期的移动平均线：ma20,ma30, 长期的有ma60
def maN(closes, day_index, N=5):
    if day_index>len(closes):
        raise Exception

    if day_index<N-1:
        return sum(closes[:day_index+1])/(day_index+1)
    else:
        return sum(closes[(day_index+1-N):day_index+1]) / N


    #usage
    '''
    data = pd.read_csv('/home/mars/Data/finialData/000032.csv')
    #求均线
    close = data['close'][:20]
    ma3 = [maN(close, i, N=3) for i in range(len(close))]
    ma5 = [maN(close, i, N=5) for i in range(len(close))]
    plt.plot(range(0,len(close)), ma3, label='ma3')
    plt.plot(range(0,len(close)), ma5, label='ma5')
    plt.legend()
    plt.show()
    '''

'''
data = pd.read_csv('/home/mars/Data/finialData/000032.csv')
#求均线
close = data['close'][:100]
# ma5 = [maN(close, i, N=5) for i in range(len(close))]
# ma10 = [maN(close, i, N=10) for i in range(len(close))]
# ma30 = [maN(close, i, N=30) for i in range(len(close))]
plt.plot(range(4,len(close)), get_maN(close, N=5), label='ma5')
plt.plot(range(9,len(close)), get_maN(close, N=10), label='ma10')
plt.plot(range(29,len(close)), get_maN(close, N=30), label='ma30')
plt.legend()
plt.show()
'''

