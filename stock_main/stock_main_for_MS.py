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
from algoruthm.unsupervised_Learning import Mean_shift_mars as ms
import copy
from sklearn.model_selection import train_test_split


'''
该模块的方法主要是针对ｄａｔａｘ做ｐｃａ　－－　ｌｏｆ处理的数据进行ｒｆ的预测
'''



def get_randomForestData(daySpan=0, code=None):
    # 获得股票的数据
    code_sql = 'SELECT * from stock_info where stock_code='+code+' and date < "2018-12-15" order by date asc;'
    code_delete_list = ['id', 'stock_code', 'stock_name', 'turnover']
    codeData = deal_dataFrame(code_sql, code_delete_list)
    final_index = codeData.index.tolist()
    print('start')
    # 获得沪深３００的数据
    st_300_sql = 'SELECT * from global_info where industry_name = "沪深300" order by date asc;'
    st_300_delete_list = ['id', 'category_name', 'industry_name', 'industry_key', 'total_money']
    stData = deal_dataFrame(st_300_sql, st_300_delete_list).loc[final_index]
    # 获得电子信息的板块的数据
    industry_sql = 'SELECT * from industry_info where industry_name = "电子信息" order by date asc;'
    industry_delete_list = ['id', 'category_name', 'industry_name', 'industry_key', 'total_money']
    indusData = deal_dataFrame(industry_sql, industry_delete_list).loc[final_index]
    # 纳克达斯数据
    # 因为美股的数据比沪深股市的数据提前，所以用美股的前一天的数据预测来预测对应的中国股市当天的数据
    # 我们在取数据的时候将美国的股市的时间提前一天，对应中国的股市的时间
    NDX_sql = 'SELECT open,close,low,high,volume,other,change_rate, DATE_ADD(date,INTERVAL 1 DAY) as date from global_info where industry_name = "纳斯达克" order by date asc;'
    #NDX_delete_list = ['id', 'category_name', 'industry_name', 'industry_key', 'total_money']
    # 对于ｎａｎ的值进行向前填充
    NDXData = deal_dataFrame(NDX_sql, []).loc[final_index].fillna(method='pad')
    # 数据整合
    # caller.join(other, lsuffix='_caller', rsuffix='_other')
    marge1 = codeData.join(stData, lsuffix='_codeData', rsuffix='_stData')
    marge2 = indusData.join(NDXData, lsuffix='_indusData', rsuffix='_NDXData')

    all_data = pd.concat([marge1, marge2], axis=1)
    all_data = all_data.dropna(axis=0)[-376:]

    # 数据对Ｙ值处理
    s_all_data_x = rf.dataX_from_dataFrame(all_data)
    s_all_data_y = rf.dataY_from_dataFrame(all_data)
    s_all_data = (s_all_data_x, s_all_data_y)
    # 特征筛选
    #final_data = Filter(use=False).Correlation_coefficient(data=s_all_data, k=150)
    #final_data = Filter(use=False).Variance_selection(data=s_all_data, threshold=53)
    # 特征选择
    delete_feature = Filter(use=False).feature_RandomForest(deal_result=all_data, final_data=s_all_data,
                                                            data_y=s_all_data_y,
                                                            cicle=100, remove_number=1, threshold=None,
                                                            daySpan=daySpan)
    t_deal_result = all_data.drop(labels=delete_feature, axis=1)
    if daySpan == 0:
        final_data_X = dataX_from_dataFrame(t_deal_result)
    else:
        final_data_X = deal_data_for_Nmean(t_deal_result, N=daySpan)
    final_data = (final_data_X, s_all_data_y)
    '''
    使用遗传算法选择模型
    data_x = final_data[0]
    data_y = final_data[1]
    ratio = 0.7
    x_train = data_x[:int(len(data_x) * ratio)]
    print(x_train.shape)
    x_test = data_x[int(len(data_x) * ratio):]
    y_train = data_y[:int(len(data_y) * ratio)]
    y_test = data_y[int(len(data_y) * ratio):]
    tpot = TPOTClassifier(verbosity=2, max_time_mins=40, config_dict="TPOT light", population_size=100, mutation_rate=0.9,
                          crossover_rate=0.1, n_jobs=-1)
    tpot.fit(x_train.astype(float), y_train.astype(float))
    print(tpot.score(x_test.astype(float), y_test.astype(float)))
    '''

    '''
    画feature_imports plot
    '''
    print('最佳特征值测试模型：')
    print('')
    feature_importances = np.copy(rf.random_forest(final_data))



    # temp_features = np.mat(feature_importances)
    # marix_feature = temp_features.T * temp_features
    # plt.matshow(marix_feature, cmap=plt.cm.hot)
    # plt.title("Pixel importances with forests of trees")
    # plt.show()


'''
直接使用处理好的ＰＣＡ数据，利用替换的方式进行奇异值的处理,没有对Ｙ值处理
'''
def fit_randomForest_MS(daySpan=0, dataPath="", stock_code=''):
    # data = pd.read_csv(dataPath)
    # data = data[::-1]
    # print(data[:10])
    # # 加入其他的指标
    # result = get_other_indicators(data)
    # deal_result = result.dropna(axis=0)
    # 利用ＬＯＦ处理原始数据进行重新的决策
    #final_data = deal_data_from_dataFrame(deal_result)

    # 根据stock_code获得相关性矩阵对应的相关性数据
    #stock_code = '\'300017'
    collection = pd.read_csv('/home/mars/Data/finialData/code_correlation.csv', index_col=0)
    collection = collection.sort_values(stock_code, ascending=False)

    #print(collection)

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
        ##print('code:', code[1:])
        path = '/home/mars/Data/finialData/electronic_infomation/' + code[1:] + '.csv'
        code_data = pd.read_csv(path, index_col='date')

        result = get_other_indicators(code_data)
        # 数据整合
        dataList.append(result)
    # 按照时间对接，并且去掉NAN数据
    df = pd.concat(dataList, axis=1)
    # pandas会 按照文件的index索引来进行重新的拼接
    new_df = df.sort_index()
    #print('new_df:', new_df[:5])

    new_df.dropna(axis=0, inplace=True)
    #print('new_df2:', new_df.get('price_change'))
    print('all shape:', new_df.shape)
    #new_df.to_csv('300017_conbine.csv')
    deal_result = new_df

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

    all_y = s_deal_data[1]
    MSx_train, MSx_test, MSy_train, MSy_test = ms.getMS_repx_data(final_data_x, all_y)
    all_x = np.vstack((MSx_train, MSx_test))
    all_y = np.concatenate((MSy_train, MSy_test), axis=0)
    max_score = fit_randomForest_rep(data=(all_x, all_y))
    print('综上的最高得分为:', max_score)




'''
直接使用处理好的ＰＣＡ数据，利用去除奇异值的方法进行奇异值处理

'''

def fit_randomForest_rep(daySpan=0, data=None, stock_code=''):
    '''

    :param daySpan:
    :param data: 已经做了标准化处理
    :return:
    '''
    data_x = data[0]
    data_y = data[1]


    # 直接使用ｐｃａ数据,将100%做特异值处理，以后重新组合起来进行，随机森林的训练
    # 拿100%的数据进行ＰＣＡ
    number = len(data_y)
    error_ratio = ms.get_errorNumber() / number
    scoreInfoList = []
    global predict_y
    for i in range(31, 32, 1):
        try:
            pca_x = PCA_mars.getPcaComponent(data_x, n_components=i)
            print(pca_x.shape)

            #x_train, x_test, y_train, y_test = train_test_split(pca_x, all_y, test_size=0.3, random_state=0, shuffle=False)

            predict_y = random_forest((pca_x, data_y))
            scoreInfoList.append((getScore(), i))
        except Exception as e:
            print(e)
            break
    scoreList = []
    for one in scoreInfoList:
        score = one[0]
        scoreList.append(score)
    max_score = max(scoreList)
    max_index = scoreList.index(max_score)
    #error_ratio = scoreInfoList[max_index][1]
    components = scoreInfoList[max_index][1]
    del scoreInfoList
    del scoreList
    return (max_score, error_ratio, components)

'''
测试多个股票
'''
def fit_randomForest_mul():
    # 股票存放的集合
    path = '/home/mars/Data/finialData/electronic_infomation/'
    parents = os.listdir(path)
    # 存放不同的股票的测试分数
    scoreList = []
    # 测试文件的数目
    number = 0
    for parent in parents:
        print('开始测试>>>',parent)
        code = parent.split('.')[0]
        number += 1
        if number <= 10:
            child = os.path.join(path, parent)
            fit_randomForest_rep(daySpan=3, dataPath=child)
            #get_randomForestData(code=code)
            scoreList.append((getScore(), parent))
        else:
            break
        print(parent, '<<<测试结束')
    print(scoreList)



def get_predict_y():
    return predict_y

#fit_randomForest_mul()
fit_randomForest_MS(stock_code='\'000021')
#get_randomForestData()