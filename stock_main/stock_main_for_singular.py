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
    marge2 = indusData.join(codeData, lsuffix='_indusData', rsuffix='_codeData')

    all_data = pd.concat([marge1, marge2], axis=1)
    all_data = all_data.dropna(axis=0)

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
直接使用处理好的ＰＣＡ数据，利用替换的方式进行奇异值的处理,包括对Ｙ值处理
'''
def fit_randomForest_repXY(daySpan=0, dataPath=""):
    data = pd.read_csv(dataPath)
    # 加入其他的指标
    result = get_other_indicators(data)
    deal_result = result.dropna(axis=0)
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

    final_data_x = nr.standardized_mars(s_deal_data[0])
    print(final_data_x.shape)
    # 直接使用ｐｃａ数据,将１００％做特异值处理，随机森林的训练
    # 拿100%的数据进行ＰＣＡ
    all_y = data_y
    pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=2)
    print(pca_x.shape)
    # x_train, x_test, y_train, y_test = train_test_split(pca_x, all_y, test_size=0.3, random_state=0, shuffle=False)

    # ｘ奇异值处理
    lof_data_x = oc.LOF_PCA_for_Clustering(pca_x, isUsePCA=False)
    # y奇异值处理
    print('all_y',all_y[-10:])
    lof_data_y = oc.replace_Singular(all_y, oc.get_pred_test())
    print('lof_y', lof_data_y[-10:])
    # all_x = np.vstack((lof_data_x, x_oc.get_pred_test()test))
    print(pca_x.shape)
    # all_y = np.concatenate((y_train, y_test), axis=0)
    # print(all_x.shape, all_y.shape)
    random_forest((lof_data_x, lof_data_y))

'''
直接使用处理好的ＰＣＡ数据，利用替换的方式进行奇异值的处理,没有对Ｙ值处理
'''
def singular_add_st(daySpan=0, stock_code=''):
    # data = pd.read_csv(dataPath)
    # data = data[::-1]
    # # 加入其他的指标
    # result = get_other_indicators(data)
    # deal_result = result.dropna(axis=0)
    # 利用ＬＯＦ处理原始数据进行重新的决策
    #final_data = deal_data_from_dataFrame(deal_result)

    #stock_code = '\'600775'
    collection = pd.read_csv('/home/mars/Data/finialData/code_correlation.csv', index_col=0)
    collection = collection.sort_values(stock_code, ascending=False)

    #print(collection)

    # 提取前10的相关性股票
    top_10 = collection.index.tolist()
    top_10 = top_10[:5]
    # 获得对应的数据
    dataList = []
    code_name = []

    # 获得股票的数据
    code_sql = 'SELECT * from stock_info where stock_code=' + stock_code[1:] + ' and date < "2018-12-15" order by date asc;'
    code_delete_list = ['id', 'stock_code', 'stock_name', 'turnover']
    codeData = deal_dataFrame(code_sql, code_delete_list)
    final_index = codeData.index.tolist()
    dataList.append(codeData)
    print('start')
    # 获得沪深３００的数据
    st_300_sql = 'SELECT * from global_info where industry_name = "沪深300" order by date asc;'
    st_300_delete_list = ['id', 'category_name', 'industry_name', 'industry_key', 'total_money']
    stData = deal_dataFrame(st_300_sql, st_300_delete_list).loc[final_index]
    dataList.append(stData)
    # 获得电子信息的板块的数据
    industry_sql = 'SELECT * from industry_info where industry_name = "电子信息" order by date asc;'
    industry_delete_list = ['id', 'category_name', 'industry_name', 'industry_key', 'total_money']
    indusData = deal_dataFrame(industry_sql, industry_delete_list).loc[final_index]
    dataList.append(indusData)
    # 纳克达斯数据



    for code in top_10:
        code_name.append(code)
        if code == stock_code:
            continue
        if collection.loc[code, stock_code] < 0.6:
            continue
        code_sql = 'SELECT * from stock_info where stock_code=' + code[1:] + ' and date < "2018-12-15" order by date asc;'
        code_delete_list = ['id', 'stock_code', 'stock_name', 'turnover']
        codeData = deal_dataFrame(code_sql, code_delete_list)
        # 数据整合
        dataList.append(codeData)
    # 按照时间对接，并且去掉NAN数据
    df = pd.concat(dataList, axis=1)

    # pandas会 按照文件的index索引来进行重新的拼接
    new_df = df.sort_index()
    #print('new_df:', new_df[:5])

    new_df.dropna(axis=0, inplace=True)
    # 时间的索引
    global date_index
    date_index = new_df.index.tolist()
    #print('new_df2:', new_df.get('price_change'))
    #print('all shape:', new_df.shape)
    deal_result = new_df
    print('shape:', deal_result.shape)


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
    number = len(all_y)
    #global predict_info
    scoreInfoList = []
    for i in range(85,86,1):

        pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=i)
        print(pca_x.shape)
        #x_train, x_test, y_train, y_test = train_test_split(pca_x, all_y, test_size=0.3, random_state=0, shuffle=False)

        # 奇异值处理
        lof_data_x = oc.LOF_PCA_for_Clustering(pca_x, isUsePCA=False)
        #all_x = np.vstack((lof_data_x, x_test))
        print(pca_x.shape)
        #all_y = np.concatenate((y_train, y_test), axis=0)
        #print(all_x.shape, all_y.shape)
        predict_y = random_forest((pca_x, all_y))
        ratio_ss = len(oc.get_delete_index())/number
        scoreInfoList.append((getScore(), ratio_ss, i))

    scoreList = []
    compent = []
    for one in scoreInfoList:
        score = one[0]
        scoreList.append(score)
        compent.append(one[2])

    plt.title(stock_code + ' --- score of component')
    plt.xlabel('component')
    plt.ylabel('score')
    plt.plot(compent, scoreList,'r-o')

    max_indx = np.argmax(scoreList)  # max value index
    suit_compent = compent[max_indx]
    plt.plot(suit_compent, scoreList[max_indx], 'ks')
    show_max = '[' + str(suit_compent) + ' ' + str(round(scoreList[max_indx], 4)) + ']'
    plt.annotate(show_max, xytext=(suit_compent, scoreList[max_indx]), xy=(suit_compent, scoreList[max_indx]))

    plt.show()
    max_score = max(scoreList)
    max_index = scoreList.index(max_score)
    error_ratio  = scoreInfoList[max_index][1]
    component = scoreInfoList[max_index][2]
    del scoreInfoList
    del scoreList
    print(max_score, error_ratio, component)
    return  predict_y






'''
利用这个相关性系数构建一个模型
'''
def singular(daySpan=0, stock_code=''):
    # data = pd.read_csv(dataPath)
    # data = data[::-1]
    # # 加入其他的指标
    # result = get_other_indicators(data)
    # deal_result = result.dropna(axis=0)
    # 利用ＬＯＦ处理原始数据进行重新的决策
    #final_data = deal_data_from_dataFrame(deal_result)

    #stock_code = '\'600775'
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
        #code_name.append(code)
        # 除去相关性系数小于0.6的股票
        if collection.loc[code, stock_code] < 0.6:
            continue
        code_sql = 'SELECT * from stock_fill where stock_code=' + code[1:] + ' and date < "2018-12-15" order by date asc;'
        code_delete_list = ['id', 'stock_code', 'stock_name', 'modify']
        codeData = deal_dataFrame(code_sql, code_delete_list)
        # 数据整合
        dataList.append(codeData)
    # 按照时间对接，并且去掉NAN数据

    df = pd.concat(dataList, axis=1)

    # pandas会 按照文件的index索引来进行重新的拼接
    new_df = df.sort_index()
    #print('new_df:', new_df[:5])

    print('new_df:', new_df.shape)
    print('new_df data:', new_df[:5])

    new_df.dropna(axis=0, inplace=True)
    # 时间的索引
    global date_index
    date_index = new_df.index.tolist()
    #print('new_df2:', new_df.get('price_change'))
    #print('all shape:', new_df.shape)
    deal_result = new_df
    print('shape:', deal_result.shape)


    data_x = dataX_from_dataFrame(deal_result)

    #print('data_x shape:', data_x[:3])
    if daySpan == 0:
        #
        if data_x.shape[1] > 80:
            data_y = dataY_from_dataFrame_5(deal_result)
        else:
            # 表示没有联合股票的参与
            data_y = dataY_5_no_correlation(deal_result)
    else:

        data_y = dataY_for_Nmean(deal_result, N=daySpan)
        data_x = data_x[:len(data_y)]

    # data_y = final_data[1]
    final_data_x = nr.standardized_mars(data_x)
    print(final_data_x.shape)


    # 直接使用ｐｃａ数据,将０．７做特异值处理，以后重新组合起来进行，随机森林的训练
    # 拿100%的数据进行ＰＣＡ
    all_y = data_y
    number = len(all_y)
    #global predict_info
    scoreInfoList = []
    for i in range(6, final_data_x.shape[1]-10, 1):

        pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=i)
        print(pca_x.shape)
        #x_train, x_test, y_train, y_test = train_test_split(pca_x, all_y, test_size=0.3, random_state=0, shuffle=False)

        # 奇异值处理
        lof_data_x = oc.LOF_PCA_for_Clustering(pca_x, isUsePCA=False)
        #all_x = np.vstack((lof_data_x, x_test))
        print(pca_x.shape)
        #all_y = np.concatenate((y_train, y_test), axis=0)
        #print(all_x.shape, all_y.shape)
        predict_y = random_forest((pca_x, all_y))
        ratio_ss = len(oc.get_delete_index())/number
        scoreInfoList.append((getScore(), ratio_ss, i))

    scoreList = []
    compent = []
    for one in scoreInfoList:
        score = one[0]
        scoreList.append(score)
        compent.append(one[2])
    '''
    plt.title(stock_code + ' --- score of component')
    plt.xlabel('component')
    plt.ylabel('score')
    plt.plot(compent, scoreList,'r-o')

    max_indx = np.argmax(scoreList)  # max value index
    suit_compent = compent[max_indx]
    plt.plot(suit_compent, scoreList[max_indx], 'ks')
    show_max = '[' + str(suit_compent) + ' ' + str(round(scoreList[max_indx], 4)) + ']'
    plt.annotate(show_max, xytext=(suit_compent, scoreList[max_indx]), xy=(suit_compent, scoreList[max_indx]))

    plt.show()
    '''
    max_score = max(scoreList)
    max_index = scoreList.index(max_score)
    error_ratio  = scoreInfoList[max_index][1]
    component = scoreInfoList[max_index][2]
    del scoreInfoList
    del scoreList

    print(max_score, error_ratio, component)
    return  (max_score, error_ratio, component)

def get_date_index():
    return date_index

'''
直接使用处理好的ＰＣＡ数据，利用去除奇异值的方法进行奇异值处理

'''
def fit_randomForest_del(daySpan=0, dataPath=""):
    data = pd.read_csv(dataPath)
    # 加入其他的指标
    result = get_other_indicators(data)
    deal_result = result.dropna(axis=0)
    # 利用ＬＯＦ处理原始数据进行重新的决策
    #final_data = deal_data_from_dataFrame(deal_result)
    data_x = dataX_from_dataFrame(deal_result)
    if daySpan == 0:
        # 对X处理, Ｙ值做二分化处理
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
    all_y = data_y
    pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=2)
    print(pca_x.shape)
    # x_train, x_test, y_train, y_test = train_test_split(pca_x, all_y, test_size=0.3, random_state=0, shuffle=False)

    # 奇异值处理
    lof_data_x = oc.LOF_PCA_for_Clustering_del(pca_x, isUsePCA=False)
    dele_index = oc.get_delete_index()
    lof_data_y = np.delete(all_y, dele_index, axis=0)
    # all_x = np.vstack((lof_data_x, x_test))
    print(lof_data_x.shape)
    print('y', data_y.shape)
    # all_y = np.concatenate((y_train, y_test), axis=0)
    # print(all_x.shape, all_y.shape)
    feature_importances = np.copy(random_forest((lof_data_x, lof_data_y)))


def getstock_id():
    data = pd.read_csv('/home/mars/桌面/股票趋势预测/处理股票数据预测/collection_all.csv')
    codes = data['code'][:90]
    print(codes)
    code_list = []
    for code in codes:
        code = str(code)[1:]
        code_list.append(code)
    return code_list

'''
测试多个股票
'''
def fit_randomForest_mul():
    # 股票存放的集合
    path = '/home/mars/Data/finialData/electronic_infomation/'
    parents = os.listdir(path)


    codeList = getstock_id()
    # 存放不同的股票的测试分数
    scoreList = []
    # 测试文件的数目
    number = 0
    with open('/home/mars/桌面/股票趋势预测/处理股票数据预测/correlation_5_all.csv', 'a+') as f:
        f.write('code,best_score,error_ratio,components,\n')

        for parent in codeList:
            print('开始测试>>>',parent)
            #code = parent.split('.')[0]
            code = parent
            number += 1
            if number <= len(codeList):
                try:
                    #child = os.path.join(path, parent)
                    (max_score, error_ratio, component) = singular(daySpan=0, stock_code='\'' + code)
                    #get_randomForestData(code=code)

                except Exception as e:
                    (max_score, error_ratio, component) = (0, 0, 0)
                    print(str(e))
                f.write('\'' + code)
                f.write(',')
                f.write(str(max_score))
                f.write(',')
                f.write(str(error_ratio))
                f.write(',')
                f.write(str(component))
                f.write(',')
                f.write('\n')
                f.flush()
            else:
                break
            print(parent, '<<<测试结束')
        print(scoreList)

if __name__ == '__main__':
    #singular(stock_code='\'300324')
    fit_randomForest_mul()

'''
#fit_randomForest_mul()
predict_y = singular_randomForest_rep(stock_code='\'000021')
date_index = get_date_index()
y_date_index = date_index[1:]
x_date_index = date_index[:-1]
test_y_date_index = y_date_index[int(len(y_date_index) * 0.7):]
stock_code = '\'000021'

#predict_y = get_predict_y()

code_sql = 'SELECT * from stock_info where stock_code=' + stock_code[1:] + ' and date < "2018-12-15" order by date asc;'
code_delete_list = ['id', 'stock_code', 'stock_name', 'turnover']
codeData = deal_dataFrame(code_sql, code_delete_list)

money = 0.0
real_money = 0.0
buy_y = predict_y[-10:]
print('buy_y: ', buy_y)
buy_date_index = test_y_date_index[-10:]
print('buy_date_index: ', buy_date_index)
# 控制仓位
time = 0
# 模拟盘
round = 0
for i, date in zip(buy_y, buy_date_index):
    round = round+1
    #最后一天必须卖掉
    if round == len(buy_y):
        if time == 1:
            money = money + codeData.loc[date, 'open']
            print(str(date),"--last sell money >>", money)
    # 涨, 买入
    if i == 1:
        if time == 0:
            # 买进下一天的股票（开盘价）
            money = money - codeData.loc[date, 'open']
            print(str(date),"--buy money >>", money)
            time = 1
    # 跌,卖出
    else:
        if time == 1:
            money = money + codeData.loc[date, 'open']
            print(str(date),"--sell money >>", money)
            time = 0

# 实盘 最后一天的卖出价格减去第一天开始买入的价格
real_money = real_money + (codeData.loc[buy_date_index[-1], 'open'] - \
             codeData.loc[buy_date_index[0], 'open'])
print('money : ', money)
print('real_money: ',real_money)

'''
#get_randomForestData()