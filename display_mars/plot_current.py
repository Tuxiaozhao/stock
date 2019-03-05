#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: plot_current.py
@time: 2019/01/22
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from STA_indicators.STA_main import get_other_indicators
from tool.unique_csv import classfic_4
from algoruthm.unsupervised_Learning import Mean_shift_mars as ms
import Feature_Engineering.Normalizer as nr
from algoruthm.dimensionality_reduction import PCA_mars
import Feature_Engineering.outliner_check as oc
from algoruthm.supervision_learning.randomForest import random_forest,getScore
from sklearn.model_selection import train_test_split



# 二分类对ｘ值做处理
def dataX_from_dataFrame(data=pd.DataFrame({})):
    temp_x = []
    for indexs in data.index:
        temp_x.append(data.loc[indexs])
        #print(data.loc[indexs].values)
    # X在此不取最后一个值，为了用ｔ的值预测ｔ＋１的值
    x = np.copy(temp_x[:-1])
    return x

# 二分类对Ｙ值做处理
def dataY_from_dataFrame(data=pd.DataFrame({})):
    # 以price_change做相应的二值化处理，作为Ｙ值
    # price_change大于０的为１，小于０的为０。自此作为股票的涨跌
    y = data.get('price_change')[1:]
    y = y.iloc[:, 0]
    #print('price_change', y)
    y = np.where(y > 0, 1, 0)
    return y


# 4分类对Ｙ值做处理
def dataY_from_dataFrame_4(data=pd.DataFrame({})):
    # 以price_change做相应的二值化处理，作为Ｙ值
    # price_change大于０的为１，小于０的为０。自此作为股票的涨跌
    y = data.get('p_change')[1:]
    y = y.iloc[:, 0]
    #print('price_change', y)
    y = np.copy(list(map(classfic_4, y)))
    print(y)
    return y


def get_data(dataPath = ''):

    stock_code = '\'000032'

    # 获得数据
    collection = pd.read_csv('/home/mars/Data/finialData/code_correlation.csv', index_col=0)
    collection = collection.sort_values(stock_code, ascending=False)

    # print(collection)

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
        # print('code:', code[1:])
        path = '/home/mars/Data/finialData/electronic_infomation/' + code[1:] + '.csv'
        code_data = pd.read_csv(path, index_col='date')

        result = get_other_indicators(code_data)
        # 数据整合
        dataList.append(result)
    # 按照时间对接，并且去掉NAN数据
    df = pd.concat(dataList, axis=1, sort=False)

    # pandas会 按照文件的index索引来进行重新的拼接
    new_df = df.sort_index()
    # print('new_df:', new_df[:5])

    new_df.dropna(axis=0, inplace=True)
    # print('new_df2:', new_df.get('price_change'))
    # print('all shape:', new_df.shape)
    deal_result = new_df

    data_x = dataX_from_dataFrame(deal_result)
    data_y = dataY_from_dataFrame(deal_result)
    return (data_x, data_y)




def get_predict_data(final_data_x, data_y, n_components):
    print('***********开始测试 original ********************')
    pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=n_components)
    original_predict_y = random_forest((pca_x, data_y))
    print('original_predict_y', original_predict_y.shape)
    original_score = round(getScore(), 4)
    return (original_predict_y, original_score)


'''
 数据
 '300324', '旋极信息'
 如果数据库没有的时间点，表示该股票对应的时间停盘了
'''
#2分类的数据
# 趋势和概率
one_tendency2 = [1, 0.85]
#5分类的数据
# 趋势和概率
one_tendency5 = [2, 0.77]

#获得测试集的模拟的数据

#对应测试集的真实数据

#即将预测的数据


#即将预测的数据对应的真实数据

# 今天的数据



'''
显示数据
'''