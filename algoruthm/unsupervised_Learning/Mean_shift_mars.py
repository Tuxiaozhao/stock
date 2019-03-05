#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: Mean_shift_mars.py
@time: 2018/12/20
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from STA_indicators.STA_main import get_other_indicators
from sklearn.model_selection import train_test_split
import Feature_Engineering.Normalizer as nr
from Feature_Engineering.Filter import Filter
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import copy
from algoruthm.dimensionality_reduction import PCA_mars
import Feature_Engineering.outliner_check as oc


'''
假设在一个多维空间中有很多数据点需要进行聚类，Mean Shift的过程如下：

1、在未被标记的数据点中随机选择一个点作为中心center；

2、找出离center距离在bandwidth之内的所有点，记做集合M，认为这些点属于簇c。
同时，把这些求内点属于这个类的概率加1，这个参数将用于最后步骤的分类

3、以center为中心点，计算从center开始到集合M中每个元素的向量，将这些向量相加，得到向量shift。

4、center = center+shift。即center沿着shift的方向移动，移动距离是||shift||。

5、重复步骤2、3、4，直到shift的大小很小（就是迭代到收敛），记住此时的center。
注意，这个迭代过程中遇到的点都应该归类到簇c。

6、如果收敛时当前簇c的center与其它已经存在的簇c2中心的距离小于阈值，那么把c2和c合并。
否则，把c作为新的聚类，增加1类。

6、重复1、2、3、4、5直到所有的点都被标记访问。

7、分类：根据每个类，对每个点的访问频率，取访问频率最大的那个类，作为当前点集的所属类。

简单的说，mean shift就是沿着密度上升的方向寻找同属一个簇的数据点。

'''

# #############################################################################

def deal_data_from_dataFrame(data=pd.DataFrame({})):
    y = data.get('price_change')[1:]
    y = np.where(y > 0, 1, 0)
    temp_x = []
    for indexs in data.index:
        temp_x.append(data.loc[indexs])
        #print(data.loc[indexs].values)

    x = np.copy(temp_x)
    return (x, y)


# Generate sample data
# centers = [[1, 1], [-1, -1], [1, -1]]
# X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)



def test():
    data = pd.read_csv('/home/mars/Data/finialData/electronic_infomation/000021.csv')
    data = data[::-1]
    result = get_other_indicators(data)


    #result = data[['price_change', 'p_change']]
    deal_result = result.dropna(axis=0)
    close = deal_result['close']
    print(close.shape)
    s_deal_data = deal_data_from_dataFrame(deal_result)
    data_x = s_deal_data[0]
    data_y = s_deal_data[1]
    # 特征处理
    #t_deal_data_x = Filter(use=False).Variance_selection(threshold=3, data=s_deal_data)[0]
    # 归一化
    final_data_x = nr.standardized_mars(data_x)

    pca_x = oc.LOF_PCA_for_Clustering(final_data_x)

    final_data_x_LOF = oc.replace_Singular(final_data_x, oc.get_pred_test())
    print('final_data_x_LOF',final_data_x_LOF[:16])

    print(final_data_x_LOF.shape)
    #降维处理
    #pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=0.9)
    # #############################################################################
    # Compute clustering with MeanShift
    x_train = final_data_x_LOF[:int(len(data_x) * 0.7)]
    print('x_train', x_train.shape)
    x_test = final_data_x_LOF[int(len(data_x) * 0.7):]
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(x_train, quantile=0.2, random_state=1)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False)
    ms.fit(final_data_x_LOF)
    labels = ms.labels_
    print('error size', labels[labels != 0].size)
    print('index of not 0 *******')
    print ([i for i, x in enumerate(labels) if x != 0])
    print('*******')
    print(labels)
    print(labels.shape)
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    #score = metrics.silhouette_score(pca_x, labels, metric='euclidean')
    #score1 = metrics.calinski_harabaz_score(pca_x, labels)
    #print(score)
    #print(score1)

    print("number of estimated clusters : %d" % n_clusters_)
    plt.plot(range(len(close)), close)
    plt.plot(range(len(labels)), labels)
    plt.show()
    # #############################################################################
    # Plot result
    '''
    from itertools import cycle
    
    plt.figure(1)
    plt.clf()
    
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(final_data_x[my_members, 0], final_data_x[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    '''

def getMS_del_data(data_x, data_y):
    pca_x = PCA_mars.getPcaComponent(data_x, n_components=0.9)

    old_x_train, old_x_test, old_y_train, old_y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0, shuffle=False)
    # #############################################################################
    # Compute clustering with MeanShift
    x_train, x_test, y_train, y_test = train_test_split(pca_x, data_y, test_size=0.3, random_state=0, shuffle=False)
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(x_train, quantile=0.2, random_state=1)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False)
    ms.fit(x_train)
    predict = ms.predict(x_test)
    labels = ms.labels_
    print(labels.shape)
    print(predict.shape)
    print('labels;', labels)
    print('predice:', predict)
    global error_number
    error_number = labels[labels != 0].size + predict[predict != 0].size
    print('error size', labels[labels != 0].size)
    print('predict error size', predict[predict != 0].size)
    print('index of not 0 *******')
    train_nomal_index = [i for i, x in enumerate(labels) if x == 0]
    print('*******')
    #print(labels)
    print('labels:', labels.shape)

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print('n_clusters_', n_clusters_)

    MSx_train = old_x_train[train_nomal_index]
    MSy_train = old_y_train[train_nomal_index]

    test_nomal_index = [i for i, x in enumerate(predict) if x == 0]
    MSx_test = old_x_test[test_nomal_index]
    MSy_test = old_y_test[test_nomal_index]

    return (MSx_train, MSx_test, MSy_train, MSy_test)


def getMS_repx_data(data_x, data_y):
    pca_x = PCA_mars.getPcaComponent(data_x, n_components=0.9)

    old_x_train, old_x_test, old_y_train, old_y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0, shuffle=False)
    # #############################################################################
    # Compute clustering with MeanShift
    x_train, x_test, y_train, y_test = train_test_split(pca_x, data_y, test_size=0.3, random_state=0, shuffle=False)
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(x_train, quantile=0.2, random_state=1)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False)
    ms.fit(x_train)
    predict = ms.predict(x_test)
    labels = ms.labels_
    global error_number
    error_number = labels[labels != 0].size + predict[predict != 0].size
    #替换出现训练集处出现特别聚类的X值
    deal_train_x = replace_Cluster(old_x_train, labels)

    deal_test_x = replace_Cluster(old_x_test, predict)

    return (deal_train_x, deal_test_x, old_y_train, old_y_test)



# 替换特别的聚类的X
def replace_Cluster(data_x, predict):
    '''

    :param pca_x: pac分析以后输出的X的值
    :param pred_test: 进行LOF分析预测的结果
    :return: 对奇异值处理以后的Ｘ
        '''
    error_size = predict[predict != 0].size
    print('error size', error_size)
    print('occpy ratio:', error_size/predict.size)
    global error_index
    error_index = [i for i, x in enumerate(predict) if x != 0]
    test = list(copy.copy(predict))
    x = np.copy(data_x)
    if isinstance(x.shape, tuple):
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

def get_errorNumber():
    return error_number

