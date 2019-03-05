#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: STA_main.py
@time: 2018/12/09
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from stockstats import *


'''
获得所有的指标
'''
def get_other_indicators(data=pd.DataFrame({})):
    '''

    :param data: 数据库，或的ｃｓｖ的数据，一定要是dataFrame的格式
    :return: dataFrame
    '''
    stock = StockDataFrame.retype(data)


    stock.get('volume_delta')

    #CR指标
    cr = stock.get('cr')
    cr_ma1 = stock.get('cr-ma1')
    cr_ma2 = stock.get('cr-ma2')
    cr_ma3 = stock.get('cr-ma3')

    #KDJ
    kdjk = stock.get('kdjk')
    kdjj = stock.get('kdjj')
    kdjd = stock.get('kdjd')

    #MACD
    macd = stock['macd']
    # MACD signal line
    macds = stock['macds']
    # MACD histogram
    macdh = stock['macdh']


    #bolling, including upper band and lower band
    boll = stock['boll']
    boll_ub = stock['boll_ub']
    boll_lb = stock['boll_lb']

    count = stock['cr-ma2_xu_cr-ma1_20_c']

    # 6 days RSI
    RSI_6 = stock['rsi_6']
    # 12 days RSI
    RSI_12 = stock['rsi_12']

    # 10 days WR
    wr_10 = stock['wr_10']
    # 6 days WR
    wr_6 = stock['wr_6']

    #CCI, default to 14 days
    cci = stock['cci']
    # 20 days CCI
    cci_20 = stock['cci_20']


    #TR (true range)
    tr = stock['tr']
    # ATR (Average True Range)
    atr = stock['atr']

    #DMA, difference of 10 and 50 moving average
    dma = stock['dma']


    # DMI
    # +DI, default to 14 days
    pdi = stock['pdi']
    # -DI, default to 14 days
    mdi = stock['mdi']
    # DX, default to 14 days of +DI and -DI
    dx = stock['dx']
    # ADX, 6 days SMA of DX, same as stock['dx_6_ema']
    adx = stock['adx']
    # ADXR, 6 days SMA of ADX, same as stock['adx_6_ema']
    adxr = stock['adxr']



    # TRIX, default to 12 days
    trix = stock['trix']
    # MATRIX is the simple moving average of TRIX
    trix_9 = stock['trix_9_sma']

    # VR, default to 26 days
    vr = stock['vr']
    # MAVR is the simple moving average of VR
    vr_6 = stock['vr_6_sma']
    return data

# 画图
'''
# plt.plot(range(len(close)), K_90,'r', label='K_90')
# plt.plot(range(len(close)), _d80,'r+', label='d_80')

#plt.plot(range(len(close)), close, 'o', label='close')
fig, (ax1, ax2)= plt.subplots(1, 2, sharex=True)

ax1.plot(range(len(close)),dma, 'b--', label='dma')
#ax1.plot(range(len(close)), atr,'r', label='atr')



ax2.plot(range(len(close)), close, 'r', label='close')
ax1.legend()
ax2.legend()
plt.show()
'''

#stock.columnName_window_statistics