#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: stock_main_for_part1.py
@time: 2018/12/28
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import algoruthm.supervision_learning.randomForest as rf
from tpot import TPOTClassifier
from algoruthm.supervision_learning.randomForest import *
from Feature_Engineering.Filter import Filter
import os, sys
from algoruthm.dimensionality_reduction import PCA_mars
import Feature_Engineering.Normalizer as nr
import Feature_Engineering.outliner_check as oc
import datetime as datetime
from sklearn.model_selection import train_test_split


'''
根据对应的股票和日期，获得对应的日期的所有相关的信息
'''




'''
直接使用处理好的ＰＣＡ数据，利用替换的方式进行奇异值的处理,没有对Ｙ值处理
由于我们的训练的数据是来自于csv文件，但是我们的展示的数据是来自于数据库
'''
def get_predict_data(daySpan=0, stock_code='', date='now', dataFrom='csv', n_components=20,
                              classfic=2):
    collection = pd.read_csv('/home/mars/Data/finialData/code_correlation.csv', index_col=0)
    collection = collection.sort_values(stock_code, ascending=False)

    # 提取前10的相关性股票
    top_10 = collection.index.tolist()
    top_10 = top_10[:5]
    # 获得对应的数据
    dataList = []
    code_name = []
    for code in top_10:
        code_name.append(code)
        # df[(df.BoolCol==3)&(df.attr==22)].index.tolist()
        # code = code_relation[code_relation.get(stock_code)==score].index
        #print('code:', code[1:])
        if dataFrom == 'csv':
            path = '/home/mars/Data/finialData/electronic_infomation/' + code[1:] + '.csv'
            code_data = pd.read_csv(path, index_col='date')
            code_data = code_data[::-1]
            #print(code_data)
            result = get_other_indicators(code_data)
        elif dataFrom == 'db':
            code_sql = 'SELECT * from stock_info where stock_code=' + code[1:] + ' order by date asc;'
            code_delete_list = ['id', 'stock_code', 'stock_name', 'turnover']
            result = deal_dataFrame(code_sql, code_delete_list)
            # 获得需要删除的行索引

        else:
            pass

        #result = get_other_indicators(code_data)
        # 数据整合
        dataList.append(result)
    # 按照时间对接，并且去掉NAN数据
    df = pd.concat(dataList, axis=1)

    # pandas会 按照文件的index索引来进行重新的拼接
    new_df = df.sort_index()
    #print('new_df:', new_df[:5])

    new_df.dropna(axis=0, inplace=True)
    #print('new_df2:', new_df.get('price_change'))
    #print('all shape:', new_df.shape)
    global now_df
    # 获得特定的行，获得以后在元数据中删除
    date_index = new_df.index.tolist()
    if isinstance(date_index[0], str):
        try:
            now_index = date_index.index('2018-12-17')
            delete_index_list = date_index[now_index:]
            now_df = pd.DataFrame(new_df, index=[date])
            new_df.drop(index=delete_index_list, inplace=True)
        except Exception as e:
            print(str(e))

    else:
        try:
            now_index = date_index.index(datetime.date(2018, 12, 17))
            delete_index_list = date_index[now_index:]
            date = datetime.datetime.strptime(date, '%Y-%m-%d')
            now_df = pd.DataFrame(new_df, index=[date])
            new_df.drop(index=delete_index_list, inplace=True)
        except Exception as e:
            print(str(e))

    print(date_index)
    try:
        pass
        # 如果给点的预测的日期小于'2018-12-14'，只需要获得改天的数据即可
        # if date_index.index('2018-12-14') < date_index.index(date):
        #     now_index = date_index.index('2018-12-17')
        #     delete_index_list = date_index[now_index:]
        #     now_df = pd.DataFrame(new_df, index=date)
        #     new_df.drop(index=delete_index_list, inplace=True)
        # else:
        #     try:
        #         now_index = date_index.index('2018-12-17')
        #         delete_index_list = date_index[now_index:]
        #         new_df.drop(index=delete_index_list, inplace=True)
        #     except Exception as e:
        #         print(str(e))
        #     now_df = pd.DataFrame(new_df, index=[date])
    except Exception:

        return '本天该股票休市'

    print('new_df:', new_df)
    print('now_df:', now_df)
    deal_result = new_df
    data_x = dataX_from_dataFrame(deal_result)

    if classfic == 2:
        if daySpan == 0:
            data_y = dataY_from_dataFrame(deal_result)
        else:
            data_y = dataY_for_Nmean(deal_result, N=daySpan)
            data_x = data_x[:len(data_y)]
    elif classfic == 5:
        if daySpan == 0:
            data_y = dataY_from_dataFrame_5(deal_result)
        else:
            data_y = dataY_for_Nmean(deal_result, N=daySpan)
            data_x = data_x[:len(data_y)]
    else:
        pass

    final_data_x = nr.standardized_mars(data_x)
    print(final_data_x.shape)
    # 直接使用ｐｃａ数据,将０．７做特异值处理，以后重新组合起来进行，随机森林的训练
    # 拿100%的数据进行ＰＣＡ
    pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=n_components)
    #x_train, x_test, y_train, y_test = train_test_split(pca_x, all_y, test_size=0.3, random_state=0, shuffle=False)
    predict_y = random_forest((pca_x, data_y))
    print('模型的分数: ', getScore())
    return predict_y


'''
获得输入的日期对应的数据
'''
def get_now_df():
    return now_df

#fit_randomForest_mul()
get_predict_data(stock_code='\'600775', date='2018-11-28', dataFrom='db')
#get_randomForestData()