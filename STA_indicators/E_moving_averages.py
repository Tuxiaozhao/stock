#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: E_moving_averages.py
@time: 2018/11/27
"""
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

'''
see the blog
https://blog.csdn.net/daodan988/article/details/51258676/
计算公式
其公式为：
EMAtoday=α * Pricetoday + ( 1 - α ) * EMAyesterday;
其中，α为平滑指数，一般取作2/(N+1)。在计算MACD指标时，EMA计算中的N一般选取12和26天，因此α相应为2/13和2/27。
当公式不断递归，直至EMA1出现，EMA1是没有定义的。EMA1 的取值有几种不同的方法，通常情况下取EMA1为Price1，
另外有的技术是将EMA1取值为开头4到5个数值的均值。
在计算机递推计算时，可以写作：
EMAtoday=α * ( Pricetoday - EMAyesterday ) + EMAyesterday;
'''

def emaN(closes, index, N=12):
    '''

    :param closes:收盘价
    :param index: 所需要获得close中的对应的下标的emaN,从0开始
    :param N:
    :return:emaN
    '''
    a = 2/(N+1)
    if index == 0:
        return closes[0]
    else:
        return a*closes[index]+(1-a)*emaN(closes, index-1, N)

'''
DIF=今日EMA（12）－今日EMA（26）
'''
def DIF(closes, index):
    ema12 = emaN(closes, index, 12)
    ema26 = emaN(closes, index, 26)
    return ema12 - ema26

'''
计算DIF的9日EMA
根据离差值计算其9日的EMA，即离差平均值，是所求的MACD值。为了不与指标原名相混淆，此值又名
DEA或DEM。
今日DEA（MACD）=前一日DEA×8/10+今日DIF×2/10。
计算出的DIF和DEA的数值均为正值或负值。
用（DIF-DEA）×2即为MACD柱状图。
故MACD指标是由两线一柱组合起来形成，快速线为DIF，慢速线为DEA，柱状图为MACD
'''
def DEA(closes, index):
    '''

    :param closes:
    :param index: 所需要获得close中的对应的下标的DEA,从0开始
    :return:
    '''
    if index == 0:
        return 0
    else:
        return 0.8*DEA(closes, index-1)+0.2*DIF(closes, index)


'''
用（DIF-DEA）×2即为MACD柱状图。
故MACD指标是由两线一柱组合起来形成，快速线为DIF，慢速线为DEA，柱状图为MACD
'''
def bar_MCDA(closes, index):
    return round((DIF(closes, index) - DEA(closes, index))*2, 4)





data = pd.read_csv('/home/mars/Data/finialData/000032.csv')
    #求均线
close = data['close'][:20]
close1 = [55.01, 53.7]
#print(close)
result = DEA(close, 12)

print(result)
print(bar_MCDA(close, 1))
