#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: MACD_points.py
@time: 2018/11/27
"""
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
import STA_indicators.E_moving_averages as ema

class MACD_points():

    def __init__(self):
        pass

    '''
    黄金交叉点， DIF线自下而上穿过DEA线
    '''
    def glodan_cross(self, closes, index):
        '''

        :param closes:
        :param index: 判断是否在这个点是否是黄金交叉点 index(对应closes的坐标)>1
        :return:
        '''
        last_DIF = ema.DIF(closes, index-1)
        last_DEA = ema.DEA(closes, index-1)
        if last_DIF < last_DEA:
            if ema.DIF(closes, index) >= ema.DEA(closes, index):
                return  True

    '''
        死亡交叉点， DIF线自上而下穿过DEA线
        '''

    def death_cross(self, closes, index):
        '''

        :param closes:
        :param index: 判断是否在这个点是否是死亡交叉点 index(对应closes的坐标)>1
        :return:
        '''
        last_DIF = ema.DIF(closes, index - 1)
        last_DEA = ema.DEA(closes, index - 1)
        if last_DIF > last_DEA:
            if ema.DIF(closes, index) <= ema.DEA(closes, index):
                return True

    '''
        上穿零轴， DIF线自下而上穿过0轴
        '''

    def up_zero_cross(self, closes, index):
        '''

        :param closes:
        :param index: 判断是否在这个点是否是0轴 index(对应closes的坐标)>1
        :return:
        '''
        last_DIF = ema.DIF(closes, index - 1)
        if last_DIF < 0:
            if ema.DIF(closes, index) >= 0:
                return True

    '''
            下穿零轴， DIF线自上而下穿过0轴
            '''

    def down_zero_cross(self, closes, index):
        '''

        :param closes:
        :param index: 判断是否在这个点是否是0轴点 index(对应closes的坐标)>1
        :return:
        '''
        last_DIF = ema.DIF(closes, index - 1)
        if last_DIF > 0:
            if ema.DIF(closes, index) <= 0:
                return True

    '''
    柱线改向, MACD能量柱从大于0 变成 小于0 或者 从小于0 变成大于0
    '''
    def MACD_change_director(self, closes, index):
        last_bar = ema.bar_MCDA(closes, index-1)
        if last_bar >= 0:
            # 看跌
            if ema.bar_MCDA(closes, index)<0:
                return (True, 0)
        else:
            # 看涨
            if ema.bar_MCDA(closes, index)>0:
                return (True, 1)
