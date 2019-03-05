#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: Svm_Mars.py
@time: 2019/01/04
"""
print(__doc__)

from sklearn import datasets, svm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stock_former.get_mian_indictors import *
from STA_indicators.STA_main import get_other_indicators
from Feature_Engineering.Filter import Filter
import algoruthm.supervision_learning.randomForest as rf
from tpot import TPOTClassifier
from algoruthm.supervision_learning.randomForest import *
from Feature_Engineering.Filter import Filter
import os, sys
from algoruthm.dimensionality_reduction import PCA_mars
import Feature_Engineering.Normalizer as nr
import Feature_Engineering.outliner_check as oc
import copy
from sklearn.model_selection import train_test_split



# 二分类对Ｙ值做处理
def deal_y(data=pd.DataFrame({})):
    # 以price_change做相应的二值化处理，作为Ｙ值
    # price_change大于０的为１，小于０的为０。自此作为股票的涨跌
    y = data.get('price_change')[1:]
    y = np.where(y > 0, 1, 0)
    return y

def fit_SVM(daySpan=0, code=None):
    stock_code = '\'000021'
    collection = pd.read_csv('/home/mars/Data/finialData/code_correlation.csv', index_col=0)
    collection = collection.sort_values(stock_code, ascending=False)

    print(collection)

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
        print('code:', code[1:])
        path = '/home/mars/Data/finialData/electronic_infomation/' + code[1:] + '.csv'
        code_data = pd.read_csv(path, index_col='date')

        result = get_other_indicators(code_data)
        # 数据整合
        dataList.append(result)
    # 按照时间对接，并且去掉NAN数据
    df = pd.concat(dataList, axis=1)

    # pandas会 按照文件的index索引来进行重新的拼接
    new_df = df.sort_index()
    print('new_df:', new_df[:5])

    new_df.dropna(axis=0, inplace=True)
    print('new_df2:', new_df.get('price_change'))
    print('all shape:', new_df.shape)
    deal_result = new_df
    # 利用ＬＯＦ处理原始数据进行重新的决策
    #final_data = deal_data_from_dataFrame(deal_result)
    data_x = dataX_from_dataFrame(deal_result)
    if daySpan == 0:
        #
        data_y = dataY_from_dataFrame(deal_result)
    else:

        data_y = dataY_for_Nmean(deal_result, N=daySpan)
        data_x = data_x[:len(data_y)]
    s_deal_data = (data_x, data_y)

    # data_y = final_data[1]
    final_data_x = nr.standardized_mars(s_deal_data[0])
    print(final_data_x.shape)


    # 直接使用ｐｃａ数据,将０．７做特异值处理，以后重新组合起来进行，随机森林的训练
    # 拿100%的数据进行ＰＣＡ
    all_y = s_deal_data[1]
    scoreListInfo = []
    for i in range(6, 40, 1):
        pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=i)
        print(pca_x.shape)
        #x_train, x_test, y_train, y_test = train_test_split(pca_x, all_y, test_size=0.3, random_state=0, shuffle=False)

        # 奇异值处理
        lof_data_x = oc.LOF_PCA_for_Clustering(pca_x, isUsePCA=False)
        #all_x = np.vstack((lof_data_x, x_test))
        print(pca_x.shape)
        x_train, x_test, y_train, y_test = train_test_split(lof_data_x, all_y, test_size=0.3, random_state=0, shuffle=False)

        # fit the model
        #for fig_num, kernel in enumerate(('linear', 'rbf', 'poly','sigmoid')):
        for c in np.arange(0.1, 1, 0.1):
            clf = svm.SVC(gamma=c, kernel='rbf')
            clf.fit(x_train, y_train)
            score = clf.score(x_test, y_test)
            print(score)
            scoreListInfo.append((score, i, c))
    #print(scoreListInfo)
    scoreList = []
    for one in scoreListInfo:
        score = one[0]
        scoreList.append(score)
    max_score = max(scoreList)
    max_index = scoreList.index(max_score)
    # error_ratio = scoreInfoList[max_index][1]
    components = scoreListInfo[max_index][1]
    c = scoreListInfo[max_index][2]
    del scoreListInfo
    del scoreList
    print('best paramers:')
    print(max_score, c, components)
    return (max_score, c, components)


fit_SVM()