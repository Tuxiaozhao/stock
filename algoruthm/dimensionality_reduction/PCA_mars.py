#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: PCA_mars.py
@time: 2018/12/22
"""
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from STA_indicators.STA_main import get_other_indicators
import Feature_Engineering.Normalizer as nr



def deal_data_from_dataFrame(data=pd.DataFrame({})):
    y = data.get('price_change')[1:]
    y = np.where(y > 0, 1, 0)
    temp_x = []
    for indexs in data.index:
        temp_x.append(data.loc[indexs])
        #print(data.loc[indexs].values)

    x = np.copy(temp_x[:-1])
    return (x, y)


'''
利用ＰＣＡ主成成分分析进行降维，获得比较优质的特征值
'''
def getPcaComponent(data_x, ratio=0.7, n_components=0.9):
    '''

    :param data_x: 数据集ｘ
    :param ratio: 使用的训练集的比率
    :param n_components: int, float, None or string
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
        MLE is used to guess the dimension. Use of ``n_components == 'mle'``
        will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.

        If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
        number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.

        If ``svd_solver == 'arpack'``, the number of components must be
        strictly less than the minimum of n_features and n_samples.

        Hence, the None case results in::

            n_components == min(n_samples, n_features) - 1
    :return: 降维后的X集
    '''
    #x_train, x_test, y_train, y_test = train_test_split(pca_x, all_y, test_size=0.3, random_state=0, shuffle=False)
    if ratio >= 1.0:
        x_train = data_x[:-1]
    else:
        x_train = data_x[:int(len(data_x) * ratio)]
        #print('x_train', x_train.shape)
    #x_test = data_x[int(len(data_x) * ratio):]
    # 降维, 指定主成分的方差和所占的最小比例阈值
    pca = PCA(n_components=n_components, random_state=0, copy=True)
    pca.fit(x_train)
    #print('各维度的方差: ', pca.explained_variance_)
    #print('各维度的方差值占总方差值的比例: ', pca.explained_variance_ratio_)
    print('占总方差值90%的维度数量: ', pca.n_components_, '\n')
    Pca_x = pca.fit_transform(data_x)
    return Pca_x



if __name__ == "__33main__":
    data = pd.read_csv('/home/mars/Data/finialData/electronic_infomation/000948.csv')
    data = data[::-1]

    result = get_other_indicators(data)
    #result = result[['open', 'close', 'low', 'high', 'volume', 'price_change']]
    deal_result = result.dropna(axis=0)
    s_deal_data = deal_data_from_dataFrame(deal_result)
    # 划分训练集和测试集，将数据集的70%划入训练集，30%划入测试集
    #train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3, random_state=1)
    # 归一化
    final_data_x = nr.standardized_mars(s_deal_data[0])
    ratio = 0.7
    data_x = final_data_x
    x_train = data_x[:int(len(data_x) * ratio)]
    print('x_train', x_train.shape)
    x_test = data_x[int(len(data_x) * ratio):]
    m, n = np.shape(x_train)


    # 降维, 指定主成分的方差和所占的最小比例阈值
    pca = PCA(n_components=0.9, random_state=1)
    pca.fit(x_train)
    print ('各维度的方差: ', pca.explained_variance_)
    print ('各维度的方差值占总方差值的比例: ', pca.explained_variance_ratio_)
    print ('占总方差值90%的维度数量: ', pca.n_components_, '\n')
    data_x = pca.fit_transform(final_data_x)

    # # 降维, 使用MLE算法计算降维后维度数量
    # pca = PCA(n_components='mle', svd_solver = 'full')
    # pca.fit(x_train)
    # print ('各维度的方差: ', pca.explained_variance_)
    # print ('各维度的方差值占总方差值的比例: ', pca.explained_variance_ratio_)
    # print ('降维后的维度数量: ', pca.n_components_, '\n')