#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: KDJ_index.py
@time: 2018/11/27
"""
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

'''
　　KDJ指标，中文名随机指标，由乔治·莱恩（GeorgeLane）创立，是目前期货和股票市场上常用的技术分析指标。
KDJ指标在设计过程中主要是研究最高价、最低价和收盘价之间的关系，体现市场情绪，可以用来测度超买或超卖现象，被广泛应用于中短期趋势分析中。

　　一般说来，K线超过90意味着超买，K线低于10意味着超卖；D线超过80意味着超买，D线低于20意味着超卖；K线在低位上穿D线为“金叉”是买入信号，K线在高位下穿D线为“死叉”是卖出信号。由此可见KDJ是非常简单实用的技术指标。

　　计算KDJ首先要计算周期的RSV值，即未成熟随机指标值，然后再依次计算K值、D值及J值。以KDJ日线数据的计算为例，其计算公式为：

'''

'''
RSV = (当日的收盘价-最近N日内的最低价)/(最近N日内的最高价-最近N日内的最低价)*100
n = 9
'''
def RSV( closes, highs, lowers, index, N=9):
    '''

    :param closes:
    :param highs:
    :param lowers:
    :param index: 对应的数组中的下标---也就是获得对应的RSV 的日期的下标
    :param N:
    :return:
    '''
    if index < N:
        highest = max(highs[:index])
        lowest = min(lowers[:index])
    else:
        highest = max(highs[index-N+1:index])
        lowest = min(highs[index - N + 1:index])
    return  round((closes[index]-lowest)/(highest-lowers)*100, 4)

'''
当日的K = 2/3*前一天的K值 + 1/3 当日的RSV的值
'''
def K_value(closes, highs, lowers, index, N=9):
    if index == 0:
        return 0
    else:
        return (2/3)*K_value(closes, highs, lowers, index-1, N)+(1/3)*RSV(closes, highs, lowers, index, N)



'''
当日的D = 2/3*前一天的D值 + 1/3 当日的K的值
'''
def D_value(closes, highs, lowers, index, N=9):
    if index == 0:
        return 0
    else:
        return (2/3)*D_value(closes, highs, lowers, index-1, N)+(1/3)*K_value(closes, highs, lowers, index, N)


'''
当日的J = 3*当日的K值 - 2* 当日的D的值
'''
def J_value(closes, highs, lowers, index, N=9):
    if index == 0:
        return 0
    else:
        return 3*K_value(closes, highs, lowers, index, N) - (2*D_value(closes, highs, lowers, index, N))

