#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: Get_sigular_time.py
@time: 2019/01/05
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stock_former.get_mian_indictors import *
from STA_indicators.STA_main import get_other_indicators
from Feature_Engineering.Filter import Filter
import algoruthm.supervision_learning.randomForest as rf
from tpot import TPOTClassifier
from Feature_Engineering.Filter import Filter
import os, sys
from algoruthm.dimensionality_reduction import PCA_mars
import Feature_Engineering.Normalizer as nr
import Feature_Engineering.outliner_check as oc
import copy
from sklearn.model_selection import train_test_split

# 二分类
def deal_data_from_dataFrame(data=pd.DataFrame({})):
    y = data.get('price_change')[1:]
    y = np.where(y > 0, 1, 0)
    temp_x = []
    for indexs in data.index:
        temp_x.append(data.loc[indexs])
        #print(data.loc[indexs].values)

    x = np.copy(temp_x[:-1])
    return (x, y)

def compare_s_no(dataPath=""):
    data = pd.read_csv(dataPath)
    # 加入其他的指标
    result = get_other_indicators(data)
    deal_result = result.dropna(axis=0)[-100:]
    # 利用ＬＯＦ处理原始数据进行重新的决策
    final_data = deal_data_from_dataFrame(deal_result)
    # data_y = final_data[1]
    final_data_x = nr.standardized_mars(final_data[0])
    print(final_data_x.shape)
    # 直接使用ｐｃａ数据,将１００％做特异值处理，随机森林的训练
    # 拿100%的数据进行ＰＣＡ
    data_y = final_data[1]
    data_x = final_data[0]
    final_data_x = nr.standardized_mars(data_x)
    # 拿100%的数据进行ＰＣＡ
    pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=54)

    # 奇异值处理
    oc.LOF_PCA_for_Clustering(pca_x, isUsePCA=False)

    #random_forest((lof_data_x, new_all_y))

    lof_pred = oc.get_pred_test()
    error_index = oc.get_delete_index()
    lof_data_y = oc.replace_Singular(data_y, lof_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    ax1.scatter(range(len(data_y)), data_y, label='data_y')
    error_close = data_y[error_index]
    # ax1.plot(range(len(lof_pred)), lof_pred, label='lof_pred')

    ax1.scatter(error_index, error_close, label='error_y', c='r', alpha=0.2)
    # ax1.xlabel('x -')
    # ax1.ylabel('y -')
    # ax1.title('plot open')
    ax1.legend()
    # ax2.ylabel('close')

    error_lof_y = lof_data_y[error_index]

    ax2.scatter(range(len(lof_data_y)), lof_data_y, label='lof_data_y')
    ax2.scatter(error_index, error_lof_y, label='error_lof_y', c='r', alpha=0.2)
    # ax2.plot(close**2, label='quadratic')
    ax2.legend()
    # 调整cavas 的间隔
    print(len(data_y))
    print(len(lof_data_y))
    plt.tight_layout()
    plt.show()


def dataX_from_dataFrame(data=pd.DataFrame({})):
    temp_x = []
    for indexs in data.index:
        temp_x.append(data.loc[indexs])
        #print(data.loc[indexs].values)
    # X在此不取最后一个值，为了用ｔ的值预测ｔ＋１的值
    x = np.copy(temp_x)
    return x
'''
直接对csv文件进行奇异值时间提取
'''
def analyze_lof(dataPath=""):
    data = pd.read_csv(dataPath)
    data = data[::-1]
    # 加入其他的指标
    result = get_other_indicators(data)
    deal_result = result.dropna(axis=0)
    # 利用ＬＯＦ处理原始数据进行重新的决策
    # final_data = deal_data_from_dataFrame(deal_result)

    # 获得电子信息的板块的数据
    # NDX_sql = 'SELECT open,close,low,high,volume,other,change_rate, DATE_ADD(date,INTERVAL 1 DAY) as date from global_info where industry_name = "纳斯达克" order by date asc;'
    # NDX_delete_list = ['id', 'category_name', 'industry_name', 'industry_key', 'total_money']
    # 对于ｎａｎ的值进行向前填充
    # NDXData = deal_dataFrame(NDX_sql, [])

    final_data = deal_data(deal_result)
    # data_y = final_data[1]
    final_data_x = nr.standardized_mars(final_data[0])

    pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=62)
    print(pca_x.shape)
    # x_train, x_test, y_train, y_test = train_test_split(pca_x, all_y, test_size=0.3, random_state=0, shuffle=False)

    # ｘ奇异值处理
    lof_data_x = oc.LOF_PCA_for_Clustering(pca_x, isUsePCA=False)
    error_index = oc.get_delete_index()
    print(error_index)
    result = deal_result.index.tolist()
    # 写入所有的日期，奇异值存在的标志为1
    with open('300113_data.csv', 'w+') as f:
        f.write('date')
        f.write(',')
        f.write('300113_Sigular')
        f.write('\n')
        for index, date in enumerate(result):
            if index in error_index:
                f.write(date)
                f.write(',')
                f.write('1')
                f.write('\n')
            else:
                f.write(date)
                f.write(',')
                f.write('0')
                f.write('\n')


def deal_data(data):
    y = data.get('change')[1:]
    #print('change:', y[-10:])
    y = np.where(y > 0, 1, 0)
    temp_x = []
    for indexs in data.index:
        temp_x.append(data.loc[indexs])
        # print(data.loc[indexs].values)

    x = np.copy(temp_x[:-1])
    return (x, y)

'''
直接对数据库数据进行奇异值时间提取
'''
def analyze_lof_sql(code=""):

    # 获得电子信息的板块的数据
    NDX_sql = 'SELECT open,close,low,high,volume,other,change_rate, DATE_ADD(date,INTERVAL 1 DAY) as date from global_info where industry_name = "纳斯达克" order by date asc;'
    # NDX_delete_list = ['id', 'category_name', 'industry_name', 'industry_key', 'total_money']
    # 对于ｎａｎ的值进行向前填充
    NDXData = deal_dataFrame(NDX_sql, [])

    final_data = deal_data(NDXData)
    # data_y = final_data[1]
    final_data_x = nr.standardized_mars(final_data[0])

    pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=62)
    print(pca_x.shape)
    # x_train, x_test, y_train, y_test = train_test_split(pca_x, all_y, test_size=0.3, random_state=0, shuffle=False)

    # ｘ奇异值处理
    lof_data_x = oc.LOF_PCA_for_Clustering(pca_x, isUsePCA=False)
    error_index = oc.get_delete_index()
    print(error_index)
    result = NDXData.index.tolist()
    # 写入所有的日期，奇异值存在的标志为1
    with open('NDX_data.csv', 'w+') as f:
        f.write('date')
        f.write(',')
        f.write('NDX_Sigular')
        f.write('\n')
        for index, date in enumerate(result):
            if index in error_index:
                f.write(date.strftime('%Y-%m-%d'))
                f.write(',')
                f.write('1')
                f.write('\n')
            else:
                f.write(date.strftime('%Y-%m-%d'))
                f.write(',')
                f.write('0')
                f.write('\n')

'''
测试多个股票
'''


# fit_randomForest_mul()
#analyze_lof(dataPath='/home/mars/Data/finialData/electronic_infomation/002362.csv')
# get_randomForestData()
analyze_lof(dataPath = '/home/mars/Data/finialData/electronic_infomation/300113.csv')