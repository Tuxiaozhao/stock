#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: K_lines.py
@time: 2018/11/26
"""
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# 所有的K线的单日指标
# 所有的k线的指标的基准是以开盘价为标准



class K_lines(object):

    def __init__(self, open, close, high, lower):
        pass

    # 大阳线
    def sun_line(self, open=0., close=0.):
        if (close-open) >= (open%0.05):
            return True

    # 大阴线
    def yin_line(self, open=0., close=0.):
        if (open-close) >= (open%0.05):
            return True

    # 十字线
    def reticle(self, open=0., close=0.):
        if abs(close-open)<=(open*0.008):
            return True
        else:
            return False

    #一字线
    def one_word_line(self,price_array):
        new_price_array = price_array[1:-1]
        new_price_array = price_array[-1:]+price_array[:-1]
        print(new_price_array)
        diff = [(abs(x-y)<=y*0.011) for x, y in zip(new_price_array, price_array)]
        print(diff)
        if False in diff:
            return False
        else:
            return True

    #T字线
    def T_line(self, open, close, high, lower):
        if abs(close-open)<=(open*0.009) and abs(high-close)<=(open*0.012) and\
                abs(close-lower)>=(open*0.02):
            return True
        else:
            return False


    #倒T字线
    def re_T_line(self, open, close, high, lower):
        if abs(close-open)<=(open*0.009) and abs(close-lower)<=(open*0.012) and\
                abs(high-close)>=(open*0.02):
            return True
        else:
            return False

    # 锤头线
    # K线的实体很小，下影线大于等于实体的两倍，上影线很短
    def hammer_line(self, open, close, high, lower):
        temp = max(open, close)
        if (high-temp) <= (open*0.006):
            #黑色
            if (open-close)>=(open*0.015):
                if (close-lower)>=(open-close)*2:
                    return [0, True]
            #白色
            elif (close-open)>=(open*0.015):
                if (open-lower)>=(close-open)*2:
                    return [1, True]
            else:
                return False
        else:
            return False


    # 流星线
    # K线的实体很小，上影线大于等于实体的两倍，下影线很短
    def meteor_line(self, open, close, high, lower):
        temp = min(open, close)
        if (temp - lower) <= (open * 0.006):
            # 黑色
            if (open - close) >= (open * 0.015):
                if (high - open) >= (open - close) * 2:
                    return [0, True]
            # 白色
            elif (close - open) >= (open * 0.015):
                if (high - close) >= (close - open) * 2:
                    return [1, True]
            else:
                return False
        else:
            return False

