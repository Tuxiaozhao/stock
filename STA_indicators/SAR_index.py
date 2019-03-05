#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: SAR_index.py
@time: 2018/12/07
"""

import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from stock_former.get_mian_indictors import *

'''
【计算公式】

1.画SAR之前，首先要决定你开始画的第一天，是属于多头或空头趋势？

2.如果第一天属于多头，则第一天的SAR一定是4天来的最低点（包括今天在内）。

3.找出开始第一天的SAR之后，紧接着计算下一日的SAR：

下一日的SAR=第一天SAR+（0·02*XP）

XP=第一天的最高点—第一天的SAR

4.第二天收盘时，又可以计算出第三天的SAR，如果第二天最高价〉第一天最高价，则第三天的SAR=第二天SAR+（0·04*XP）

XP=第二天的最高点—第二天的SAR

只要最高价〉前一日最高价，则其乘数因子一律增加0·02，如果一直累增下去，最高只能累增至0·2为止，之后一律以0·2为乘数因子。

如果最高价≤前一日最高价，则第三天的SAR和第二天的SAR相同，而且乘数因子也不必累加。

第三天收盘后，依上述步骤持续在每日价格下方画出SAR，直到某一天收盘价跌破SAR，
则代表行情转为空头应卖出股票，而行情转为空头的当天，
立即将四天来的最高点，做为次一日的空头SAR。

5.反转后第二天的最低点如果≥前一天的最低点，则SAR和前一天相同。

注意！SAR虽然和前一天相同，也必须在图形上画出来。

6.反转后第二天的最低点若〈前一天的最低点，则

第三天的SAR=第二天的SAR+90·02*XK）。

XK=第二天的最低价—第二天的SAR。

第三天以后持续依照上述步骤，在每日价格上方画出SAR，直到某一天收盘价向上突破SAR，代表行情为多头应买进股票，
而行情转为多头的当天，立刻按照2的步骤设定SAR。

注意！一般SAR的参数设定为四天，读者朋友们应尽量不要更改。

递推公式：

    SAR（n）＝SAR（n-1）＋AF[EP（n-1）－SAR（n-1）]

    SAR（n）＝第n日的SAR值，SAR（n-1）即第（n-1）日之值；

AR；加速因子；

EP：极点价，若是看涨一段期间，则EP为这段期间的最高价，若是看跌一段时间，则EP为这段期间的最低价：EP（n-1）：第（n-1）日的极点价。

'''

def get_SAR(close, high, low, N=4):
    '''

    :param close:
    :param high:
    :param low:
    :param N:
    :return: 预测的是第I+1天内的价格， I天的价格是已知的（也就是今天的价格已知）
    '''
    # 得到周期的第一天的sar
    #今天的SAR是由前面一天决定的（和前3天决定）
    sar_upper = []
    sar_down = []
    result = []

    ap = 0.02
    for i in range(len(close)):
        # flag_upper = 0
        # flag_low = 0
        if i >= N-1:
            # 首先判断是上涨还是下跌
            if len(sar_upper) == 0 and len(sar_down) == 0:
                if close[i]>close[i-N+1]:
                    # if flag_low == 1:
                    #     ap = 0.02
                    # flag_upper = 1
                    # flag_low = 0
                    sar_1 = min(low[i-N+1: i+1])
                    sar_upper.append(sar_1)
                else:
                    # if flag_upper == 1:
                    #     ap = 0.02
                    # flag_upper = 0
                    # flag_low = 1
                    sar_1 = max(high[i - N + 1: i+1])
                    sar_down.append(sar_1)

            # 计算上涨的行情
            elif len(sar_upper) > 0:

                if close[i] > sar_upper[-1]:
                    if len(sar_upper) == 1:
                        # 计算第二天的sar
                        # 下一日的SAR=第一天SAR+（0·02*XP）
                        # XP=第一天的最高点—第一天的SAR
                        xp_1 = high[i] - sar_upper[-1]
                        sar_2 = sar_upper[-1]+(ap*xp_1)
                        sar_upper.append(sar_2)
                    else:
                        # 计算第三天的sar
                        # 如果第二天最高价〉第一天最高价，则第三天的SAR=第二天SAR+（0·04*XP）
                        # XP=第二天的最高点—第二天的SAR

                        if high[i] > high[i - 1]:
                            if ap >= 0.2:
                                ap = 0.2
                            else:
                                ap = ap + 0.02

                            xp = high[i] - sar_upper[-1]
                            sar_3 = sar_upper[-1] + (ap * xp)

                        # 如果最高价≤前一日最高价，则第三天的SAR和第二天的SAR相同，而且乘数因子也不必累加。
                        else:
                            sar_3 = sar_upper[-1]
                        sar_upper.append(sar_3)
                #直到某一天收盘价跌破SAR，
                #则代表行情转为空头应卖出股票，而行情转为空头的当天，
                #立即将四天来的最高点，做为次一日的空头SAR。
                else:
                    sar_down.append(max(high[i - N + 1: i+1]))
                    result.extend(sar_upper)
                    sar_upper = []
                    ap = 0.02

            # 计算下跌趋势
            elif len(sar_down) > 0:
                if close[i] < sar_down[-1]:
                    if len(sar_down) == 1:
                        # 计算第二天的sar
                        # 下一日的SAR=第一天SAR+（0·02*XP）
                        # XP=第一天的最地点—第一天的SAR
                        xp_1 = low[i] - sar_down[-1]
                        sar_2 = sar_down[-1] + (ap * xp_1)
                        sar_down.append(sar_2)
                    else:
                        # 计算第三天的sar
                        # 反转后第二天的最低点若〈前一天的最低点，则
                        # 第三天的SAR=第二天的SAR+（0·02*XK）。
                        # #XK=第二天的最低价—第二天的SAR。
                        if low[i] < low[i - 1]:
                            if ap >= 0.2:
                                ap = 0.2
                            else:
                                ap = ap + 0.02
                            xp = low[i] - sar_down[-1]
                            sar_3 = sar_down[-1] + (ap * xp)

                        # 反转后第二天的最低点如果≥前一天的最低点，则SAR和前一天相同。，而且乘数因子也不必累加。
                        else:
                            sar_3 = sar_down[-1]
                        sar_down.append(sar_3)

                else:
                    #直到某一天收盘价向上突破SAR，代表行情为多头应买进股票，
                    #而行情转为多头的当天，立刻按照2的步骤设定SAR。
                    sar_upper.append(min(low[i - N + 1: i+1]))
                    result.extend(sar_down)
                    sar_down = []
                    ap = 0.02
    result.extend(sar_down)
    result.extend(sar_upper)
    return  result


# close  = [1,2,3,4,5,6]
# high  = [1,2,3,4,5,6]
# low  = [1,2,3,4,5,6]

info = getdata_by_code('002446',[])

data = info[-100:]
np_data = np.copy(data)
close = np_data[:, 5]
high = np_data[:, 4]
low = np_data[:, 6]
print('close',close)
print('high',high)
print('low', low)



result = get_SAR(close, high, low)

plt.plot(close,'b--', label='close')
plt.plot(high,'y', label='high')
plt.plot(low, 'g' ,label='low')
print(len(result))
plt.plot(range(4, len(close)+1),result, 'r', label='SAR')
plt.legend()
plt.show()











'''






def get_SAR(df, N=4, step=2, maxp=20):
    sr_value = []
    sr_up = []
    ep_up = []
    af_up = []
    sr_down = []
    ep_down = []
    af_down = []
    for i in range(len(df)):
        if i >= N:
            if len(sr_up) == 0 and len(sr_down) == 0:
                if df.ix[i, 'close'] > df.ix[0, 'close']:
                    # 标记为上涨趋势
                    sr0 = df['low'][0:i].min() # 默认的sr0
                    af0 = 0.02
                    ep0 = df.ix[i, 'high']
                    sr_up.append(sr0)
                    ep_up.append(ep0)
                    af_up.append(af0)
                    sr_value.append(sr0)
                if df.ix[i, 'close'] <= df.ix[0, 'close']:
                    # 标记为下跌趋势
                    sr0 = df['high'][0:i].max()
                    af0 = 0.02
                    ep0 = df.ix[i, 'high']
                    sr_down.append(sr0)
                    ep_down.append(ep0)
                    af_down.append(af0)
                    sr_value.append(sr0)
            if len(sr_up) > 0:
                if df.ix[i - 1, 'low'] > sr_up[-1]:
                    sr0 = sr_up[-1]
                    ep0 = df['high'][-len(sr_up):].max()
                    if df.ix[i, 'high'] > df['high'][-(len(sr_up) - 1):].max():
                        af0 = af_up[-1] + 0.02
                    if df.ix[i, 'high'] <= df['high'][-(len(sr_up) - 1):].max():
                        af0 = af_up[-1]

                    sr = sr0 + af0 * (ep0 - sr0)
                    sr_up.append(sr)
                    ep_up.append(ep0)
                    af_up.append(af0)
                    sr_value.append(sr)
                    print('上涨sr0={},ep0={},af0={},sr={}'.format(sr0, ep0, af0, sr))
                if df.ix[i - 1, 'low'] <= sr_up[-1]:
                    ep0 = df['high'][-len(sr_up):].max()
                    sr0 = ep0
                    af0 = 0.02
                    sr_down.append(sr0)
                    ep_down.append(ep0)
                    af_down.append(af0)
                    sr_value.append(sr0)
                    sr_up = []
                    ep_up = []
                    af_up = []
            if len(sr_down) > 0:
                if df.ix[i - 1, 'high'] < sr_down[-1]:
                    sr0 = sr_down[-1]
                    ep0 = df['low'][-len(sr_down):].max()
                    if df.ix[i, 'low'] < df['low'][-(len(sr_down) - 1):].max():
                        af0 = af_down[-1] + 0.02
                    if df.ix[i, 'low'] >= df['low'][-(len(sr_down) - 1):].max():
                        af0 = af_down[-1]

                    sr = sr0 + af0 * (ep0 - sr0)
                    sr_down.append(sr)
                    ep_down.append(ep0)
                    af_down.append(af0)
                    sr_value.append(sr)
                    print('下跌sr0={},ep0={},af0={},sr={}'.format(sr0, ep0, af0, sr))
                if df.ix[i - 1, 'high'] >= sr_down[-1]:
                    ep0 = df['low'][-len(sr_up):].max()
                    sr0 = ep0
                    af0 = 0.02
                    sr_up.append(sr0)
                    ep_up.append(ep0)
                    af_up.append(af0)
                    sr_value.append(sr0)
                    sr_down = []
                    ep_down = []
                    af_down = []
    return sr_value

'''