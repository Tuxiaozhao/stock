#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:mars
@file: get_mian_indictors.py
@time: 2018/12/02
"""
import numpy as np

# import matplotlib.pyplot as plt
import pandas as pd
import dataBase.Connection as conn
from STA_indicators.STA_main import get_other_indicators

'''
使用 apriori 算法 选择相关的指标
'''
'''
从数据库获得一只股票的数据
'''

#delete_tuple = ('id', 'stock_code', 'stock_name', 'close', 'date', 'turnover')
indicators = []


def getdata(stock_code, delete_tuple):
    # 默认连接本地数据库
    con = conn.mysql_operator()
    # 深科技 >.> 000021
    sql = "SELECT * from stock_info where stock_code='"+stock_code+"'order by date asc;"
    result = con.select_stock_info(sql)
    data_list = []
    index = 0
    for one in result:
        temp = []
        index += 1
        for k, v in one.items():

            if k not in delete_tuple:
                if index == 1:
                    indicators.append(k)
                temp.append(v)
        data_list.append(temp)


    return data_list



def getdata_by_code(stock_code, delete_tuple):
    # 默认连接本地数据库
    con = conn.mysql_operator()
    # 深科技 >.> 000021
    sql = "SELECT * from stock_info where stock_code='"+stock_code+"'order by date asc;"
    result = con.select_stock_info(sql)
    data_list = []
    index = 0
    for one in result:
        temp = []
        index += 1
        for k, v in one.items():

            if k not in delete_tuple:
                if index == 1:
                    indicators.append(k)
                temp.append(v)
        data_list.append(temp)


    return data_list



'''
获得将数据库中的数据转化成的ＤＡＴＡＦＲＡＭＥ的数据
'''
def deal_dataFrame(sql, del_list):
    # 默认连接本地数据库
    con = conn.mysql_operator()
    original_data = con.get_pd_data(sql=sql)

    s_deal_data = original_data.drop(columns=del_list)
    # 充电指标
    t_deal_data = get_other_indicators(s_deal_data).dropna(axis=0)
    return t_deal_data

'''
def treate_data(data):
    final_data = []
    if len(data)<1:
        data = getdata()
    np_data = np.copy(data)
    mean_data = np.mean(np_data, axis=0)
    print(mean_data)
    #for i in np_data
    # 将数据简单化
    print(np_data)
    f = np.where([np_data >= mean_data], 1, 0)
    print(f)
    #将所有的指标对号如作
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
          if f[i, j] == 1:
              f[i, j] = (j+1)
    # 将为0 的指标去掉
    for i in range(f.shape[0]):
        temp = list(f[i, :])
        while 0 in temp:
            temp.remove(0)
        final_data.append(temp)
'''
'''
数据预处理，计算出每个指标的平均值，大于平均值的极为1 反之极为0
获得指标矩阵
'''


def treate_data(data):
    print(type(data).__name__)

    if type(data).__name__ == 'DataFrame':
        data = np.copy(data.values)

    final_data = []
    if len(data) < 1:
        data = getdata()
    np_data = np.copy(data)
    mean_data = np.mean(np_data, axis=0)
    # for i in np_data
    # 将数据简单化
    for i in range(np_data.shape[0]):
        temp = list(np_data[i, :])
        # 返回满足条件的下标,下标就是对应的指标
        index = list(np.where([temp >= mean_data])[1])
        final_data.append(index)
    return final_data


'''
总的指标，下标与指标矩阵的列对应
'''


def get_indicators():
    return indicators

#
# A = pd.DataFrame()
# #
# result = treate_data(A)
# print(result)



'''
讲所有的指标取N天的平均值，所以总的数据量会减少(N-1)/N
'''
def get_all_meanN_no(data, N=3):
    result = []
    for one in range(N,len(data)+1,N):
        temp = data[one-N:one]
        result.append(np.mean(temp, axis=0))

    return np.copy(result)


def get_all_meanN(data, N=3):
    weights = np.ones(N) / N
    sma = np.convolve(weights, data)[N - 1:-N + 1]

    return sma


# data = np.arange(0, 24).reshape((8,3))
# print(data)
# print('***')
# result = get_all_meanN(data, 2)
# print(result)