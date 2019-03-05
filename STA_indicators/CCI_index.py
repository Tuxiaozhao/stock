#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: CCI_index.py
@time: 2018/12/08
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stock_former.get_mian_indictors import *


def get_CCI(close, high, low, N=14):
    tp = pd.Series((close+high+low)/3)
    print(len(tp))
    # cci = pd.Series((tp - pd.rolling_mean(tp, N)) / (0.015 * pd.rolling_std(tp, N)),
    #                 name='CCI')

    cci = pd.Series((tp - tp.rolling(N).mean()) / (0.015 * tp.rolling(N).std()),
                    name='CCI')
    return cci

info = getdata_by_code('002446',[])
print('ok')
data = info[-200:]
np_data = np.copy(data)
close = np_data[:, 5]
high = np_data[:, 4]
low = np_data[:, 6]
result = get_CCI(close, high, low)

fig, (ax1, ax2)= plt.subplots(1, 2, sharex=True)
ax1.plot(close, label='close')
ax1.plot(high, label='high')
# ax1.xlabel('x -')
# ax1.ylabel('y -')
#ax1.title('plot open')
ax1.legend()

y1 = np.ones((len(close),))
y1[:] = 100
y2 = np.ones((len(close),))
y2[:] = -100

ax2.plot(y1,'y', label='+100')
ax2.plot(y2, 'g' ,label='-100')
ax2.plot(range(len(close)-len(result), len(close)), result, 'r', label='CCI')

ax2.legend()
# 调整cavas 的间隔
plt.tight_layout()


# plt.plot(close,'b--', label='close')
# plt.plot(100,'y', label='+100')
# plt.plot(-100, 'g' ,label='-100')
# print(result)
# plt.plot(range(len(close)-len(result), len(close)), result, 'r', label='CCI')
# plt.legend()
plt.show()
