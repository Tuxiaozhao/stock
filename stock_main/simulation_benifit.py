#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: simulation_benifit.py
@time: 2019/01/28
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stock_main.Interval_time_tendancy import get_fit_model, get_interval_indexs
from stock_former.get_mian_indictors import deal_dataFrame
import datetime as datetime
from tool.unique_csv import classfic_4
'''
利用训练好的模型进行股票的模拟操作获得对应的收益
'''


# 二分类对Ｙ值处理有相关系数的股票做处理
def dataY_from_dataFrame(data=pd.DataFrame({})):
    # 以price_change做相应的二值化处理，作为Ｙ值
    # price_change大于０的为１，小于０的为０。自此作为股票的涨跌
    y = data.get('change')
    y = y.iloc[:, 0]
    #print('price_change', y)
    y = np.where(y > 0, 1, 0)
    return y


# 二分类对Ｙ值做处理
def dataY_no_correlation(data=pd.DataFrame({})):
    # 以price_change做相应的二值化处理，作为Ｙ值
    # price_change大于０的为１，小于０的为０。自此作为股票的涨跌
    y = data.get('change')
    #print('price_change', y)
    y = np.where(y > 0, 1, 0)
    return y

def dataY_from_dataFrame_5(data=pd.DataFrame({})):
    # 以price_change做相应的二值化处理，作为Ｙ值
    # price_change大于０的为１，小于０的为０。自此作为股票的涨跌
    y = data.get('change')[1:]
    y = y.iloc[:, 0]
    #print('price_change', y)
    y = np.copy(list(map(classfic_4, y)))
    #print(y)
    return y

# 4分类对Ｙ值做处理没有相关股票
def dataY_5_no_correlation(data=pd.DataFrame({})):
    # 以price_change做相应的二值化处理，作为Ｙ值
    # price_change大于０的为１，小于０的为０。自此作为股票的涨跌
    y = data.get('change')[1:]
    #print('price_change', y)
    y = np.copy(list(map(classfic_4, y)))
    return y



if __name__ == '__main__':



    stock_code = '\'300290'
    # p1:日期，p2:时间间隔
    interval = [2018, 12, 17, 5]

    # 获得预测的日期
    #date_for_python = datetime.datetime.strptime(interval[0], '%Y-%m-%d')
    date_for_python = datetime.date(interval[0], interval[1], interval[2])
    code_sql = 'SELECT * from stock_fill where stock_code=' + stock_code[
                                                      1:] + ' order by date asc;'
    code_delete_list = ['id', 'stock_code', 'stock_name', 'modify']
    codeData = deal_dataFrame(code_sql, code_delete_list)
    date_indexes = codeData.index.tolist()
    print(date_indexes)
    # 得到给定的日期在总的数据中的索引的位置
    now_index = date_indexes.index(date_for_python)
    # 获得时间段索引
    # +2 shi
    interval_index = date_indexes[now_index:now_index+interval[3]+1]

    # 获得x索引
    x_date_index = interval_index[:-1]
    # 获得y索引
    y_date_index = interval_index[1:]



    # 获得股票对应的预测信息
    info = pd.read_csv('/home/mars/桌面/股票趋势预测/处理股票数据预测/collection_all.csv',index_col='code')
    print(info[:3])
    info = info.loc[stock_code]
    components = int(info.components)



    #score, predict_y = get_fit_model(stock_code=stock_code, interval=x_date_index, n_components=components)
    score, predict_y = get_fit_model(stock_code=stock_code, interval=x_date_index, n_components=components, classfic=5)

    money = 0.0
    real_money = 0.0
    buy_y = predict_y

    print('buy_date_index: ', y_date_index)

    true_y = pd.DataFrame(codeData, index=y_date_index)
    # if true_y.shape[1] > 80:
    #     true_y = dataY_from_dataFrame(true_y)
    # else:
    #     # 表示没有联合股票的参与
    #     true_y = dataY_no_correlation(true_y)

    if true_y.shape[1] > 80:
        true_y = dataY_from_dataFrame_5(true_y)
    else:
        # 表示没有联合股票的参与
        true_y = dataY_5_no_correlation(true_y)

    print('buy_y: ', buy_y)
    print('****')
    print('true_y: ', true_y)
    print('****')
    # 控制仓位
    time = 0
    # 模拟盘
    round = 0
    # 其中 i=1表示涨， 0 表示跌
    for i, y_date, x_date in zip(buy_y, y_date_index, x_date_index):
        round = round + 1
        # 涨, 买入
        if i == 1:
            if time == 0:
                # 买进下一天的股票（前一天的收盘价买入）
                # 前一天的收盘价高于预测的日期的最低价，则能进行买卖，反之。
                if codeData.loc[x_date, 'close'] > codeData.loc[y_date, 'low']:
                    money = money - codeData.loc[x_date, 'close']
                    print(str(x_date), "--the close money >>", codeData.loc[x_date, 'close'])
                    print(str(y_date), "--buy money >>", money)
                    time = 1


        # 跌,卖出
        else:
            if time == 1:
                # 卖不比买，卖要立刻出售，所以只能比较开盘价
                # 如果前一天的收盘价高于预测日期的开盘价，我们就按预测日期的开盘价卖出
                # 否则按照前一天的收盘价卖出
                if codeData.loc[x_date, 'close'] > codeData.loc[y_date, 'open']:
                    money = money + codeData.loc[y_date, 'open']
                    print('按照预测日期的开盘价卖出')
                    print(str(y_date), "--sell money >>", money)
                    time = 0
                else:
                    print('按照前一天的收盘价卖出')
                    money = money + codeData.loc[x_date, 'close']
                    print(str(y_date), "--sell money >>", money)
                    time = 0
        # 最后一天必须卖掉（以最后的一天的收盘价卖出）
        if round == len(buy_y):
            if time == 1:
                money = money + codeData.loc[y_date, 'close']
                print(str(y_date), "--last sell money >>", money)

    # 实盘 最后一天的卖出价格(以最后的一天的收盘价卖出)减去第一天开始买入的价格
    real_money = real_money + (codeData.loc[y_date_index[-1], 'close'] - \
                               codeData.loc[y_date_index[0], 'open'])
    print('money : ', money)
    print('real_money: ', real_money)