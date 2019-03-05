#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day_tendancy.py
@time: 2019/01/23
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


# 对特定的日期ｘ值做处理
def dataX_today(data=pd.DataFrame({})):
    temp_x = []
    for indexs in data.index:
        temp_x.append(data.loc[indexs])
        #print(data.loc[indexs].values)
    # X在此不取最后一个值，为了用ｔ的值预测ｔ＋１的值
    x = np.copy(temp_x)
    return x

'''
根据对应的股票和日期，获得对应的日期的所有相关的信息
'''
def get_today_data(data_x, today_data, n_components=20):
    add_data_x = np.vstack((data_x, today_data))
    final_data_x = nr.standardized_mars(add_data_x)
    print('final_data_x:', final_data_x.shape)
    # 直接使用ｐｃａ数据,将０．７做特异值处理，以后重新组合起来进行，随机森林的训练
    # 拿100%的数据进行ＰＣＡ
    pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=n_components)
    today_x = pca_x[-1].reshape(1, n_components)
    print('today_x:', today_x)
    print(today_x.shape)
    return today_x





'''
直接使用处理好的ＰＣＡ数据，利用替换的方式进行奇异值的处理,没有对Ｙ值处理
由于我们的训练的数据是来自于csv文件，但是我们的展示的数据是来自于数据库
'''
def get_fit_model(daySpan=0, stock_code='', today='now', dataFrom='csv', n_components=20,
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
        if collection.loc[code, stock_code] < 0.6:
            continue
        code_name.append(code)
        # df[(df.BoolCol==3)&(df.attr==22)].index.tolist()
        # code = code_relation[code_relation.get(stock_code)==score].index
        # print('code:', code[1:])
        if dataFrom == 'csv':
            path = '/home/mars/Data/finialData/electronic_infomation/' + code[1:] + '.csv'
            code_data = pd.read_csv(path, index_col='date')
            code_data = code_data[::-1]
            # print(code_data)
            result = get_other_indicators(code_data)
        elif dataFrom == 'db':
            code_sql = 'SELECT * from stock_fill where stock_code=' + code[1:] + ' order by date asc;'
            code_delete_list = ['id', 'stock_code', 'stock_name', 'modify']
            result = deal_dataFrame(code_sql, code_delete_list)
            # 获得需要删除的行索引

        else:
            pass

        # result = get_other_indicators(code_data)
        # 数据整合
        dataList.append(result)
    # 按照时间对接，并且去掉NAN数据
    df = pd.concat(dataList, axis=1)

    # pandas会 按照文件的index索引来进行重新的拼接
    new_df = df.sort_index()

    print('new_df:', new_df.shape)
    #print('new_df data:', new_df[:5])

    new_df.dropna(axis=0, inplace=True)

    global now_df
    # 获得特定的行，获得以后在元数据中删除
    date_index = new_df.index.tolist()
    if isinstance(date_index[0], str):
        try:
            # csv文件训练的样本以这作为分割
            #now_index = date_index.index('2018-12-14')
            #delete_index_list = date_index[now_index:]
            now_df = pd.DataFrame(new_df, index=[today])
            #new_df.drop(index=delete_index_list, inplace=True)
        except Exception as e:
            print(str(e))

    else:
        try:
            now_index = date_index.index(datetime.date(2018, 12, 17))
            delete_index_list = date_index[now_index:]
            date = datetime.datetime.strptime(today, '%Y-%m-%d')
            now_df = pd.DataFrame(new_df, index=[date])
            new_df.drop(index=delete_index_list, inplace=True)
        except Exception as e:
            print(str(e))
    deal_result = new_df
    today_x = dataX_today(now_df)
    #print('now_df:', now_df)

    data_x = dataX_from_dataFrame(deal_result)
    today_x2 = get_today_data(data_x, today_x, n_components=n_components)
    if classfic == 2:
        if daySpan == 0:
            if data_x.shape[1] > 80:
                data_y = dataY_from_dataFrame(deal_result)
            else:
                # 表示没有联合股票的参与
                data_y = dataY_no_correlation(deal_result)
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
    print('final_data_x:', final_data_x.shape)
    # 直接使用ｐｃａ数据,将０．７做特异值处理，以后重新组合起来进行，随机森林的训练
    # 拿100%的数据进行ＰＣＡ
    pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=n_components)
    #print('data_y: ', data_y)

    # x_train, x_test, y_train, y_test = train_test_split(pca_x, all_y, test_size=0.3, random_state=0, shuffle=False)
    predict_y, future_y = random_forest((pca_x, data_y), another_data_x=today_x2)
    print('模型的分数: ', getScore())
    print('预测今天的趋势是: ', future_y)
    del pca_x
    del final_data_x
    del today_x
    del deal_result
    del date_index
    del dataList
    return (getScore(), future_y)


'''
获得输入的日期对应的数据
'''

#fit_randomForest_mul()




#score, future_y = get_fit_model(stock_code='\'600775', today='2018-9-28', dataFrom='db')

# 根据分数匹配不同的半径, future_y 为1在




if __name__ == '__main__':

    predict_date = '2018-01-24'

    plt.ylim(-1.2, 1.2)
    plt.xlim(50, 80)
    plt.ion()
    ax = plt.gca()
    for i in range(50,85,5):
        plt.plot([i,i], [-1.2, 1.2], 'k--')

    # 去掉边界
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    plt.plot([50, 90], [0, 0])
    # 设置坐标
    plt.xlabel('accuracy of few stock model')
    plt.ylabel('prediction of '+ predict_date)

    #plt.grid(color='blue', linestyle='--', linewidth=2)

    info = pd.read_csv('/home/mars/桌面/股票趋势预测/处理股票数据预测/correlation_test.csv')
    info = info[:6]
    code = info.code
    higest_score = max(list(info.best_score))
    components = info.components


    #info  = pd.DataFrame()
    for _index in info.index:

        print(info.loc[_index].values[0])
        code = info.loc[_index].values[0]
        best_score = info.loc[_index].values[1]
        components = info.loc[_index].values[3]

        #code = '\'300324'
        #components = 80

        score, future_y = get_fit_model(stock_code=code, today=predict_date, dataFrom='db', n_components=components)
        # 最高分
        #scores.append(score)
        if score >= higest_score-0.01:
            higest_score = score
            if future_y == 1:
                y = 1.01

            else:
                y = -1.01
            max_y = y
        else:
            if future_y == 1:
                y = np.random.random()+0.1
            else:
                y = -np.random.random()-0.1

        color = np.random.random(3).reshape((1, 3))
        size = score*70
     #,s= 100,linewidths=lValue,marker='o'
        plt.scatter(100*score, y, color=color, s=70, linewidths=size, marker='o', alpha=0.6)
        plt.text(100*score, y, code)
        plt.pause(0.1)
    #



    # 标注最大的准确率股票
    #max_y = 1.01

    #higest_score = max(list(scores))
    plt.annotate(s='a max score ['+str(round(higest_score, 4))+']', xy=(100*higest_score, max_y),
                 xytext=(100*higest_score+1, max_y+0.2), arrowprops={'arrowstyle':'-|>'})

    plt.ioff()
    plt.show()
    #get_randomForestData()