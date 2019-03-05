#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: boll.py
@time: 2018/12/07
"""
"""
布林带
"""
import numpy as np
import datetime
import matplotlib.pyplot as plt
from stock_former.get_mian_indictors import *
"""

step1 :读取数据

    2017浦发银行2.CSV：
    时间,收盘价(浦发银行),最高价(浦发银行),最低价(浦发银行)
    2017-01-03,12.3804137039431,12.4867485455721,12.2816742081448
    2017-01-04,12.403199741435,12.4183904330963,12.2892695539754
    2017-01-05,12.3804137039431,12.4411764705882,12.3348416289593
    2017-01-06,12.2892695539754,12.3804137039431,12.2512928248222
    …………

"""
def bolliner(close, N=20):
    '''

    :param close:
    :param N: N日移动平均计算的布林线
    :return: (upperBB, sma, lowerBB)
    '''

    # 平均权重
    weights = np.ones(N) / N

    # 卷积实现移动平均
    sma = np.convolve(weights, close)[N - 1:-N + 1]

    deviation = []

    lenc = len(close)
    for i in range(N - 1, lenc):
        dev = close[i - N + 1:i + 1]
        deviation.append(np.std(dev))

    # 两倍标准差
    deviation = 2 * np.array(deviation)
    # 三倍标准差
    three_deviation = 2 * np.array(deviation)
    # 压力线
    upperBB = sma + deviation
    # 支撑线
    lowerBB = sma - deviation
    #三倍标准差
    super_upperBB = sma + three_deviation
    super_lowerBB = sma - three_deviation

    return (upperBB, sma, lowerBB, super_upperBB, super_lowerBB)


info = getdata_by_code('002446',[])

data = info[-80:]
np_data = np.copy(data)
close = np_data[:, 5]
result = bolliner(close)
upperBB = result[0]
sma = result[1]
lowerBB = result[2]



super_upperBB = result[3]
super_lowerBB = result[4]

plt.plot(close[20:],'y',label = "close")
plt.plot(sma,'b--',label = "BOLL")
plt.plot(upperBB,'r',label = "UPR")
plt.plot(lowerBB,'g',label = "DWN")

plt.plot(super_upperBB,'r+',label = "super_UPP")

plt.plot(super_lowerBB,'r--',label = "super_DWN")
plt.title('002446\ BOLL$')
plt.legend()
plt.show()
