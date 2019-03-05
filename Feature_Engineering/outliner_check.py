#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: outliner_check.py
@time: 2018/12/24
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import os, sys
from algoruthm.dimensionality_reduction import PCA_mars
import Feature_Engineering.Normalizer as nr
from STA_indicators.STA_main import get_other_indicators
import copy

print(__doc__)


def deal_data_from_dataFrame(data=pd.DataFrame({})):
    y = data.get('price_change')[1:]
    y = np.where(y > 0, 1, 0)
    temp_x = []
    for indexs in data.index:
        temp_x.append(data.loc[indexs])
        #print(data.loc[indexs].values)

    x = np.copy(temp_x)
    return (x, y)


def other_main():

    np.random.seed(42)
    data = pd.read_csv('/home/mars/Data/finialData/electronic_infomation/300297.csv')
    data = data[::-1]
    result = get_other_indicators(data)
    delete_feature = []
    deal_result = result.dropna(axis=0)
    # print(deal_result)
    print('***')
    #print(len(columns))

    final_data = deal_data_from_dataFrame(deal_result)
    data_y = final_data[1]
    final_data_x = nr.standardized_mars(final_data[0])
    print(final_data_x.shape)
    pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=0.9)

    # xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    # # Generate normal (not abnormal) training observations
    # X = 0.3 * np.random.randn(100, 2)
    # X_train = np.r_[X + 2, X - 2]
    # # Generate new normal (not abnormal) observations
    # X = 0.3 * np.random.randn(20, 2)
    # X_test = np.r_[X + 2, X - 2]
    # # Generate some abnormal novel observations
    # X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

    # fit the model for novelty detection (novelty=True)
    print('pca_x', pca_x.shape)
    clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
    clf.fit(pca_x)
    # DO NOT use predict, decision_function and score_samples on X_train as this
    # would give wrong results but only on new unseen data (not used in X_train),
    # e.g. X_test, X_outliers or the meshgrid
    y_pred_test = clf.predict(pca_x)
    print(y_pred_test)
    error_index = [i for i, x in enumerate(y_pred_test) if x == -1]


    print('error size', y_pred_test[y_pred_test == -1].size)
    print('index of witch is -1 *******')
    print ([i for i, x in enumerate(y_pred_test) if x == -1])
    print('*******')
    # y_pred_outliers = clf.predict(X_outliers)
    # n_error_test = y_pred_test[y_pred_test == -1].size
    # n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size


    '''
    # plot the learned frontier, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.title("Novelty Detection with LOF")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
    
    s = 40
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
                     edgecolors='k')
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                    edgecolors='k')
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.legend([a.collections[0], b1, b2, c],
               ["learned frontier", "training observations",
                "new regular observations", "new abnormal observations"],
               loc="upper left",
               prop=matplotlib.font_manager.FontProperties(size=11))
    plt.xlabel(
        "errors novel regular: %d/40 ; errors novel abnormal: %d/40"
        % (n_error_test, n_error_outliers))
    plt.show()
    
    '''

    '''
    我们认为股票是会一直保持着原来的趋势
    所以，我们将检测出来的特异值，进行替换
    替换成他前面一个非特异值.aa
    '''


def replace_Singular(pca_x, LOF_predict):
    '''

    :param pca_x: pac分析以后输出的X的值
    :param pred_test: 进行LOF分析预测的结果
    :return: 对奇异值处理以后的Ｘ
    '''
    error_size = LOF_predict[LOF_predict == -1].size
    print('error size', error_size)
    print('occpy ratio:', error_size/LOF_predict.size)
    global error_index
    error_index = [i for i, x in enumerate(LOF_predict) if x == -1]
    test = list(copy.copy(LOF_predict))
    x = np.copy(pca_x)
    if isinstance(x.shape, tuple) :
        for one_index in error_index:
            if one_index == 0:
                first_index_data = x[test.index(1)]
                x[0] = first_index_data
                test[0] = 1
            else:
                x[one_index] = x[one_index - 1]
                test[one_index] = 1
    else:
        for one_index in error_index:
            if one_index == 0:
                first_index_data = x[test.index(1), :]
                x[0, :] = first_index_data
                test[0] = 1
            else:
                x[one_index, :] = x[one_index - 1, :]
                test[one_index] = 1
        #print('x:',x[:, 1][:10])
    return x


'''
    我们认为股票是会一直保持着原来的趋势
    所以，我们将检测出来的特异值，进行删除
'''
def delete_Singular(pca_x, LOF_predict):
    '''

    :param pca_x: pac分析以后输出的X的值
    :param pred_test: 进行LOF分析预测的结果
    :return: 对奇异值处理以后的Ｘ
    '''
    error_size = LOF_predict[LOF_predict == -1].size
    print('error size', error_size)
    print('occpy ratio:', error_size/LOF_predict.size)
    global error_index
    error_index = [i for i, x in enumerate(LOF_predict) if x == -1]
    #test = list(copy.copy(LOF_predict))
    x = np.copy(pca_x)
    new_x = np.delete(x, error_index, axis=0)
    del x
    return new_x

'''
利用主成成分分析，降为后，利用ＬＯＦ　基于密度的离群点检测方法，检测出特异的行，进行替换。
'''

def LOF_PCA_for_Clustering(final_data_x, isUsePCA=True, ratio=0.7):
    '''

    :param final_data_x: 初始的进行归一化的x值　或者是已经进行ＰＣＡ处理的值
    :param isUsePCA ; 是否使用PCD进行降为
    :return:
    '''
    global pred_test
    if isUsePCA:
        pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=0.9, ratio=ratio)
        print('pca_x', pca_x.shape)
        clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
        clf.fit(pca_x)

        pred_test = clf.predict(pca_x)
        return replace_Singular(pca_x, pred_test)
    else:
        clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
        #print(final_data_x[:, 1][:10])
        clf.fit(final_data_x)
        pred_test = clf.predict(final_data_x)
        return replace_Singular(final_data_x, pred_test)

'''
利用主成成分分析，降为后，利用ＬＯＦ　基于密度的离群点检测方法，检测出特异的行，进行删除。
'''
def LOF_PCA_for_Clustering_del(final_data_x, isUsePCA=True, ratio=0.7):
    '''

    :param final_data_x: 初始的进行归一化的x值　或者是已经进行ＰＣＡ处理的值
    :param isUsePCA ; 是否使用PCD进行降为
    :return:
    '''
    global pred_test
    if isUsePCA:
        pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=0.9, ratio=ratio)
        print('pca_x', pca_x.shape)
        clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
        clf.fit(pca_x)

        pred_test = clf.predict(pca_x)
        return delete_Singular(pca_x, pred_test)
    else:
        clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
        #print(final_data_x[:, 1][:10])
        clf.fit(final_data_x)
        pred_test = clf.predict(final_data_x)
        return delete_Singular(final_data_x, pred_test)

'''
测试不同种类的测试比重
'''
def LOF_PCA_for_Clustering_more(final_data_x, isUsePCA=True, ratio_for_pca=0.7, ratio_for_lof=0.7):
    '''

    :param final_data_x: 初始的进行归一化的x值　或者是已经进行ＰＣＡ处理的值
    :param isUsePCA ; 是否使用PCD进行降为
    :return:
    '''
    global pred_test
    if isUsePCA:
        pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=0.9, ratio=ratio_for_pca)
        print('pca_x', pca_x.shape)

        if ratio_for_lof >= 1.0:
            lof_data = pca_x[:-1]
            test_x = []
        else:
            lof_data = pca_x[:int(len(pca_x) * ratio_for_lof)]
            test_x = pca_x[int(len(pca_x) * ratio_for_lof):-1]
        clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
        clf.fit(lof_data)
        pred_test = clf.predict(lof_data)
        return (replace_Singular(lof_data, pred_test), test_x)
    else:
        clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
        clf.fit(final_data_x)
        pred_test = clf.predict(final_data_x)
        return replace_Singular(final_data_x, pred_test)


def get_pred_test():
    print(pred_test[-10:])
    return pred_test

def get_delete_index():
    #print(error_index)
    return error_index


# test = np.copy([1, -1, 1, -1, -1, -1, -1, 1, 1])
# pca_x = np.arange(18).reshape(9,2)
# all_y = [1,0,1,0,1,1,1,0,0]
#
#
# print('******')
# new_x = delete_Singular(pca_x, test)
# print(pca_x)
# dele_index = get_delete_index()
# data_y = np.delete(all_y, dele_index, axis=0)
# print(dele_index)
# print(new_x)
# print(data_y)