#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: Mean_mul_lines.py
@time: 2018/11/26
"""
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import STA_indicators.Mean_line as ml


class Mean_mul_lines():

    def __init__(self):
        pass

    # 黄金交叉
    #短期移动的平均线上穿长期移动平均线
    def golden_cross(self, closes, index, short_ma_N, long_ma_N):
        #上一个点没有穿过（短期均线在长期均线的下面），但是这个点穿过
        last_short_ma = ml.maN(closes, index-1, short_ma_N)
        last_long_ma = ml.maN(closes, index-1, long_ma_N)
        if last_short_ma < last_long_ma:
            short_ma = ml.maN(closes, index, short_ma_N)
            long_ma = ml.maN(closes, index, long_ma_N)
            if short_ma >= long_ma:
                return (True, index)

    # 死亡交叉
    #长期移动的平均线下穿短期移动平均线
    def death_cross(self, closes, index, short_ma_N, long_ma_N):
        '''

        :param closes:
        :param index: 是否在该点形成死亡交叉点
        :param short_ma_N:
        :param long_ma_N:
        :return:
        '''
        #上一个点没有穿过（短期均线在长期均线的上面），但是这个点穿过
        last_short_ma = ml.maN(closes, index-1, short_ma_N)
        last_long_ma = ml.maN(closes, index-1, long_ma_N)
        if last_short_ma > last_long_ma:
            short_ma = ml.maN(closes, index, short_ma_N)
            long_ma = ml.maN(closes, index, long_ma_N)
            if short_ma <= long_ma:
                return (True, index)

    # usage
    '''
    mml = Mean_mul_lines()
    data = pd.read_csv('/home/mars/Data/finialData/000032.csv')
    #求均线
    close = data['close'][:20]
    points = [mml.golden_cross(close, i, 3, 5) for i in range(3, len(close))]
    death_points = [mml.death_cross(close, i, 3, 5) for i in range(3, len(close))]
    print(points)
    print([x[1] for i, x in enumerate(points) if (x !=None and x[0] == True) ])
    print ([x[1] for i, x in enumerate(death_points) if (x !=None and x[0] == True)  ])
    '''


    # 银山谷
    #由三根移动平均线做成,短期上穿中期和长期,中期上穿长期,从而形成一个箭头向上的不规则三角形
    # 有三个黄金交叉点,且出现的顺序有规律(先是短期均线于长期均线形成两个交叉点,后来是中期与长期成交叉点)
    # 在这三个黄金交叉点中间没有任何的其他交叉点
    #,形成三角形的三个点(这一点也是最后一个黄金交叉点)
    def silver_valley(self, closes, index, short_ma_N, middle_ma_N, long_ma_N):
        '''

        :param closes:
        :param index: 是否在该点形成银山谷
        :param short_ma_N:
        :param middle_ma_N:
        :param long_ma_N:
        :return: True/False
        LEFT : 中期的均线会不会与长期的均线产生死亡交叉???
        '''
        # 先确定这是最后一个黄金交叉点
        if self.golden_cross(closes, index, middle_ma_N, long_ma_N):
            # 银山谷一定是从 中期和长期的最近的一个死亡交叉点开始的
            print('潜在的银山谷',index)
            death_points = [self.death_cross(closes, i, short_ma_N, middle_ma_N) for i in
                             range(middle_ma_N - 1, index - 1)]
            death_points = [x[1] for i, x in enumerate(death_points) if (x != None and x[0] == True)]
            print(death_points)
            # 最近的一个中期和长期的死亡交叉点
            start_index = death_points[-1]
            print(start_index)

            #找到第一个黄金交叉点
            first_goldan = [self.golden_cross(closes, i, short_ma_N, middle_ma_N) for i in range(index-1, start_index, -1)]
            print(first_goldan)
            #second_goldan = [self.golden_cross(closes, i, short_ma_N, long_ma_N) for i in range(index-1, start_index, -1)]

            # 判断 交叉点1和3之间没有死亡交叉点
            #second_index = first_goldan.index(not None, start=-1)[1]
            first_goldan = [x[1] for i, x in enumerate(first_goldan) if (x != None and x[0] == True)]
            second_index = first_goldan[-1]
            print(second_index)
            death_points1 = [self.death_cross(closes, i, short_ma_N, middle_ma_N) for i in range(second_index, index-1)]
            print('death_points1', death_points1)
            death_points2 = [self.death_cross(closes, i, short_ma_N, long_ma_N) for i in range(second_index, index-1)]
            if (death_points1.count(None)==len(death_points1)) and (death_points2.count(None) == len(death_points2)):
                return True


    #usage
    '''
    mml = Mean_mul_lines()
    data = pd.read_csv('/home/mars/Data/finialData/000032.csv')
    #求均线
    close = data['close'][:100]
    points = [mml.silver_valley(close, i, 5, 10, 30) for i in range(1, len(close))]
    try:
        print(points.index(True)+1)
    except Exception as e:
        print('can not find')
    '''


