#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: __init__.py.py
@time: 2018/12/28
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from STA_indicators.STA_main import get_other_indicators

print(__doc__)
import Feature_Engineering.Normalizer as nr
from Feature_Engineering.Filter import Filter
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
from algoruthm.dimensionality_reduction import PCA_mars
import Feature_Engineering.outliner_check as oc
from sklearn.model_selection import train_test_split

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

    x = np.copy(temp_x[:-1])
    return (x, y)


# Generate sample data
# centers = [[1, 1], [-1, -1], [1, -1]]
# X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)


if __name__ == '__main__':

    data = pd.read_csv('/home/mars/Data/finialData/electronic_infomation/002544.csv')
    data = data[::-1]
    result = get_other_indicators(data)


    #result = data[['price_change', 'p_change']]
    deal_result = result.dropna(axis=0)
    # close = deal_result['close']
    #
    s_deal_data = deal_data_from_dataFrame(deal_result)
    data_x = s_deal_data[0]
    data_y = s_deal_data[1]
    print('data_x', data_x.shape)
    # 特征处理
    #t_deal_data_x = Filter(use=False).Variance_selection(threshold=3, data=s_deal_data)[0]
    # 归一化
    final_data_x = nr.standardized_mars(data_x)
    #
    # pca_x = oc.LOF_PCA_for_Clustering(final_data_x)
    #
    # final_data_x_LOF = oc.replace_Singular(final_data_x, oc.get_pred_test())
    # print('final_data_x_LOF',final_data_x_LOF[:16])
    #
    # print(final_data_x_LOF.shape)
    #降维处理
    pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=0.9)
    # #############################################################################
    # Compute clustering with MeanShift
    x_train, x_test, y_train, y_test = train_test_split(pca_x, data_y, test_size=0.3, random_state=0, shuffle=False)

    #x_train = pca_x[:int(len(data_x) * 0.7)]
    print('x_train', x_train.shape)
    #x_test = pca_x[int(len(data_x) * 0.7):]
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(x_train, quantile=0.2, random_state=1)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False)
    ms.fit(x_train)
    predict = ms.predict(x_test)
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

    print('777777777')
    print(predict)

#score = metrics.silhouette_score(pca_x, labels, metric='euclidean')
#score1 = metrics.calinski_harabaz_score(pca_x, labels)
#print(score)
#print(score1)

# print("number of estimated clusters : %d" % n_clusters_)
# plt.plot(range(len(close)), close)
# plt.plot(range(len(labels)), labels)
# plt.show()
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