#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: Normalizer.py
@time: 2018/12/13
"""
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from STA_indicators.STA_main import get_other_indicators

'''
标准化需要计算特征的均值和标准差，公式表达为：
x = (x-mean(x))/s
'''

def standardized_mars(data):
    deal_data = StandardScaler().fit_transform(data)
    return deal_data
'''
区间缩放法的思路有多种，常见的一种为利用两个最值进行缩放
'''

def Interval_scaling_mars(data):
    deal_data = MinMaxScaler().fit_transform(data)
    return deal_data

'''
　简单来说，标准化是依照特征矩阵的列处理数据，其通过求z-score的方法，
将样本的特征值转换到同一量纲下。归一化是依照特征矩阵的行处理数据，
其目的在于样本向量在点乘运算或其他核函数计算相似性时，
拥有统一的标准，也就是说都转化为“单位向量”。
'''
def normalized_mars(data):
    deal_data = Normalizer().fit_transform(data)
    return deal_data




def deal_data_from_dataFrame(data=pd.DataFrame({})):
    y = data.get('price_change')[1:]
    y = np.where(y > 0, 1, 0)
    temp_x = []
    for indexs in data.index:
        temp_x.append(data.loc[indexs])
        # print(data.loc[indexs].values)

    x = np.copy(temp_x[:-1])
    return (x, y)


'''
data = pd.read_csv('/home/mars/Data/002446.csv')
data = data[::-1]
result = get_other_indicators(data)
deal_result = result.dropna(axis=0)[-10:]
#print(deal_result['open'])
# print(deal_result)
final_data = deal_data_from_dataFrame(deal_result)
print(final_data[0][:, 0])
input_data = standardized_mars(final_data[0])
print(input_data[:, 0])


[-2.63443016  0.21231427  0.29030727  1.18722675 -0.25564372  0.25131077
  0.56328277  0.17331777  0.21231427]
x
a = [5.72 , 6.45, 6.47, 6.7,6.33,6.46,6.54,6.44, 6.45]

my_mean = sum(a)/len(a)
print(my_mean)
std = np.std(a)
print(std)
result = []
for one in a:

    result.append((one-my_mean)/std)

print(result)
'''

