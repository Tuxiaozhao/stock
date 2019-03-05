#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: DMI_index.py
@time: 2018/11/27
"""
# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class DMI_index(object):

    def __init__(self):
        pass

    '''
    上升动向（+DM） 
    +DM代表正趋向变动值即上升动向值，
    其数值等于当日的最高价减去前一日的最高价，如果<=0 则+DM=0。 
    '''
    def positive_DM(self, highs, index):
        if index == 0:
            return 0
        else:
            dm = highs[index] - highs[index-1]
            if dm <= 0:
                return  0
            else:
                return dm

    '''
    下降动向（-DM） 
    ﹣DM代表负趋向变动值即下降动向值，
    其数值等于前一日的最低价减去当日的最低价，如果<=0 则-DM=0。注意-DM也是非负数。 
    '''

    def minus_DM(self, lowers, index):
        if index == 0:
            return 0
        else:
            dm = lowers[index-1] - lowers[index]
            if dm <= 0:
                return 0
            else:
                return dm


    '''
    无动向 
    无动向代表当日动向值为“零”的情况，即当日的+DM和﹣DM同时等于零。有两种股价波动情况下可能出现无动向。
    一是当当日的最高价低于前一日的最高价并且当日的最低价高于前一日的最低价，二是当上升动向值正好等于下降动向值。
    '''

    def zero_DM(self, highs, lowers, index):

        if self.minus_DM(lowers, index) == self.positive_DM(highs, index):
            return True


    '''
    计算真实波幅（TR） 
    TR代表真实波幅，是当日价格较前一日价格的最大变动值。取以下三项差额的数值中的最大值（取绝对值）为当日的真实波幅： 
    A、当日的最高价减去当日的最低价的价差。 
    B、当日的最高价减去前一日的收盘价的价差。 
    C、当日的最低价减去前一日的收盘价的价差。
    TR是A、B、C中的数值最大者 
    '''

    def Tr(self, closes, highs, lowers, index):
        if index == 0:
            return abs(highs[index] - lowers[index])
        else:
            a = abs(highs[index] - lowers[index])
            b = abs(highs[index] - closes[index-1])
            c = abs(lowers[index] - closes[index-1])
        return  max(a, b, c)

    '''
    计算方向线DI 
方向线DI是衡量股价上涨或下跌的指标，分为“上升指标”和“下降指标”。在有的股市分析软件上，+DI代表上升方向线，-DI代表下降方向线。其计算方法如下： 
+DI=（+DM÷TR）×100 
-DI=（-DM÷TR）×100


    要使方向线具有参考价值，则必须运用平滑移动平均的原理对其进行累积运算。以12日作为计算周期为例，先将12日内的+DM、-DM及TR平均化，所得数值分别为+DM12，-DM12和TR12，具体如下： 
+DI（12）=（+DM12÷TR12）×100 
-DI（12）=（-DM12÷TR12）×100
    '''
    def positive_DI(self, closes, highs, lowers, index, mean=7):
        '''

        :param closes:
        :param highs:
        :param lowers:
        :param index:
        :param mean: 窗口
        :return:
        '''
        if index<mean-1:
            return 0
        else:
            sum_DM = sum([self.positive_DM(highs, index-i) for i in range(mean)])
            sum_Tr = sum([self.Tr(closes, highs, lowers, index-i) for i in range(mean)])
            return sum_DM/sum_Tr*100

    def minus_DI(self, closes, highs, lowers, index, mean=7):
        if index<mean-1:
            return 0
        else:
            sum_DM = sum([self.minus_DM(lowers, index - i) for i in range(mean)])
            sum_Tr = sum([self.Tr(closes, highs, lowers, index - i) for i in range(mean)])
            return sum_DM / sum_Tr * 100


    '''
    计算动向平均数ADX 
依据DI值可以计算出DX指标值。
其计算方法是将+DI和—DI间的差的绝对值除以总和的百分比得到动向指数DX。
由于DX的波动幅度比较大，一般以一定的周期的平滑计算，得到平均动向指标ADX。具体过程如下： 
DX=(DI DIF÷DI SUM) ×100 
其中，DI DIF为上升指标和下降指标的差的绝对值 
DI SUM为上升指标和下降指标的总和 
ADX就是DX的一定周期n的移动平均值。
    '''
    def DX(self, closes, highs, lowers, index, mean=1):

        di_dif = self.positive_DI(closes, highs, lowers, index, mean) - \
        self.minus_DI(closes, highs, lowers, index, mean)
        di_sum = self.positive_DI(closes, highs, lowers, index, mean) + \
        self.minus_DI(closes, highs, lowers, index, mean)
        return (di_dif/di_sum)*100

    def ADX(self, closes, highs, lowers, index, mean=7):
        return sum([self.DX(closes, highs, lowers, index-i) for i in range(mean)])/mean



    '''
    计算评估数值ADXR 
在DMI指标中还可以添加ADXR指标，以便更有利于行情的研判。 
ADXR的计算公式为： 
ADXR=（当日的ADX+前n日的ADX）÷2 \n为选择的周期数 
和其他指标的计算一样，由于选用的计算周期的不同，
DMI指标也包括日DMI指标、周DMI指标、月DMI指标年DMI指标以及分钟DMI指标等各种类型。
经常被用于股市研判的是日DMI指标和周DMI指标
    '''
    def ADXR(self, closes, highs, lowers, index, N=1):
        return sum([self.ADX(closes, highs, lowers, index - i) for i in range(N)]) / 2

    '''
    # dmi = DMI_index()
    # data = pd.read_csv('/home/mars/Data/finialData/000032.csv')
    #     #求均线
    # close = data['close'][:20]
    # high = data['high'][:20]
    # lower = data['low'][:20]
    # result = dmi.positive_DI(close, high, lower, 6)
    #
    # positive_points = [dmi.positive_DI(close, high, lower, i) for i in range(len(close))]
    # munis_points = [dmi.minus_DI(close, high, lower, i) for i in range(len(close))]
    #
    # plt.plot(range(0, len(close)), positive_points, label='positive_points')
    # plt.plot(range(0, len(close)), munis_points, label='munis_points')
    # plt.legend()
    # plt.show()
    '''
