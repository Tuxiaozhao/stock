#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: Filter.py
@time: 2018/12/10
"""
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from stock_former.get_mian_indictors import *
from stockstats import *
from STA_indicators.STA_main import get_other_indicators
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import *
from algoruthm.supervision_learning.randomForest import *
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

'''
特征选择

'''
class Filter(object):

    def __init__(self, filepath='/home/mars/Data/002446.csv', use=True):
        '''

        :param filepath:本身数据集的文件位置
        :param use: 是否使用本方法自带的数据集,False 表示另启数据集
        '''
        self.use = use
        if use:
            data = pd.read_csv(filepath)
            data = data[::-1]
            result = get_other_indicators(data)
            deal_result = result.dropna(axis=0)
            # print(deal_result)
            final_data = self._deal_data_from_dataFrame(deal_result)
            self.data_x = final_data[0]
            self.data_y = final_data[1]
        else:
            pass


    def _deal_data_for_Regression(self, data=pd.DataFrame({})):
        y = data.get('close')[1:]
        #y = np.where(y > 0, 1, 0)
        temp_x = []
        for indexs in data.index:
            temp_x.append(data.loc[indexs])
            #print(data.loc[indexs].values)

        x = np.copy(temp_x[:-1])
        return (x, y)


    def _deal_data_from_dataFrame(self, data=pd.DataFrame({})):
        y = data.get('price_change')[1:]
        y = np.where(y > 0, 1, 0)
        temp_x = []
        for indexs in data.index:
            temp_x.append(data.loc[indexs])
            #print(data.loc[indexs].values)

        x = np.copy(temp_x[:-1])
        return (x, y)

    '''
    This feature selection algorithm looks only at the features (X),
     not the desired outputs (y), 
    and can thus be used for unsupervised learning.
    '''
    def Variance_selection(self, threshold=3, data=None):
        '''

        :param threshold:　方差的阀值，低于该方差将会被淘汰
        :param data: 传入的是一个拖；tuple, data[0]表示x， data[1]表示Y
        :return:
        '''
        #y_train = data_y[:int(len(data_y) * ratio)]
        if self.use:

            model = VarianceThreshold(threshold).fit(self.data_x)
            new_x = model.transform(self.data_x)
            # 获得对应的选择后的特征值
            #indices = model.get_support(indices=True)
            return (new_x, self.data_y)
        else:
            model = VarianceThreshold(threshold).fit(data[0])
            new_x = model.transform(data[0])
            # 获得对应的选择后的特征值
            # indices = model.get_support(indices=True)
            return (new_x, data[1])


    '''
    使用相关系数法，先要计算各个特征对目标值的相关系数以及相关系数的P值
    See also 
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

    mutual_info_classif --  是针对最好的算法
    '''
    def Correlation_coefficient(self, func=mutual_info_regression, k=51, data=None):
        '''

        :param func: 相关的方法
        :param k: 需要选择的特征值的个数
        :param data: if use = False 则ｄａｔａ为外部的数据，否则为ＦＩｌｔｅｒ自己的数据集
        :return:
        '''
        if self.use:
            model = SelectKBest(func, k, random_state = 0).fit(self.data_x, self.data_y)
            new_x = model.transform(self.data_x)
            return (new_x, self.data_y)
        else:
            model = SelectKBest(func, k).fit(data[0], data[1])
            new_x = model.transform(data[0])
            return (new_x, data[1])

    '''
    Wrapper
    递归特征消除法
　　递归消除特征法使用一个基模型来进行多轮训练，
    每轮训练后，消除若干权值系数的特征，再基于新的特征集进行下一轮训练。
    '''
    def recursive_feature_elimination(self, n_features_to_select=50, data=None):
        #递归特征消除法，返回特征选择后的数据
        # 参数estimator为基模型
        # 参数n_features_to_select为选择的特征个数
        if self.use:
            model = RFE(estimator=tree.DecisionTreeClassifier(), n_features_to_select=n_features_to_select).fit(self.data_x, self.data_y)
            new_x = model.transform(self.data_x)
            return (new_x, self.data_y)
        else:
            model = RFE(estimator=tree.DecisionTreeClassifier(), n_features_to_select=n_features_to_select).fit(
                data[0], data[1])
            new_x = model.transform(data[0])
            return (new_x, data[1])


    '''
    基于树模型的特征选择法
　　 树模型中GBDT也可用来作为基模型进行特征选择
    '''
    def Feature_selection_bot(self, max_features=50, data=None):
        if self.use:
            model = SelectFromModel(GradientBoostingClassifier(), threshold='0.5*mean', max_features=max_features).fit(self.data_x, self.data_y)
            new_x = model.transform(self.data_x)
            return (new_x, self.data_y)
        else:
            model = SelectFromModel(GradientBoostingClassifier(), threshold='0.5*mean', max_features=max_features).fit(
                data[0], data[1])
            new_x = model.transform(data[0])
            return (new_x, data[1])


    '''
    利用随机森林做特征选择
    '''
    def feature_RandomForest(self, deal_result=None, final_data=None, data_y=None, cicle=10, remove_number=2, threshold=None, daySpan=0):
        '''

        :param deal_result: 去掉nan的数据
        :param final_data: 模型训练的最终处理数据
        :param data_y: 数据集Ｙ
        :param cicle: 循环的次数
        :param remove_number: 每次循环去掉的特征值的数目
        :param threshold: 初始时去掉特征值所占的ｆeature importance的比例，默认是初始特征的个数的倒数
        :param daySpan: ==0 表示不做时间跨度的处理，　否则表示做daySpan天的时间处理
        :return:需要最佳去除的特征值
        '''
        score_features = []
        feature_importances = np.copy(random_forest(final_data))
        delete_feature = []
        # 初始的特征值不用删除，

        score_features.append((getScore(), np.copy(delete_feature)))

        columns = list(deal_result.columns.values)
        # print(feature_importances)
        number = len(feature_importances)
        # 除掉特征贡献小于　1 / number　的特征
        if threshold == None:
            threshold = 1 / number

        for i in range(number):
            if feature_importances[i] < threshold:
                # delect_index.append(i)
                delete_feature.append(columns[i])

        # 去掉贡献很小的features
        for round in range(1, cicle+1):
            # 被删除的特征值不能少于原数量的１／３
            if len(delete_feature) <= int(number * 11 / 12):
                if delete_feature == []:
                    break
                else:
                    # 去掉贡献值小的特征值
                    print('开始第',str(round),'轮处理')
                    deal_result_2 = deal_result.drop(labels=delete_feature, axis=1)
                    columns_2 = list(deal_result_2.columns.values)
                    print(len(columns_2))
                    final_data_2_X = dataX_from_dataFrame(deal_result_2)
                    final_data_2_X = final_data_2_X[:len(data_y)]
                    final_data_2 = (final_data_2_X, data_y)
                    feature_importances_2 = random_forest(final_data_2)
                    score_features.append((getScore(), np.copy(delete_feature)))
                    # print(feature_importances_2)
                    # 讲贡献值最小的两位排除
                    for i in range(remove_number):
                        min_index = list(feature_importances_2).index(min(feature_importances_2))
                        # 讲其从元素组移除
                        feature_importances_2 = np.delete(feature_importances_2, min_index)
                        delete_feature.append(columns_2[min_index])
                    del (deal_result_2)
                    del (columns_2)
                    del (final_data_2_X)
                    del (feature_importances_2)
            else:
                print('特征数量太少，为')
                break
        # 比较得出最高的分数，并且输出对应的特征值
        scoreList = []
        for one in score_features:
            score = one[0]
            scoreList.append(score)

        max_socre = max(scoreList)
        max_index = scoreList.index(max_socre)
        remove_feature = score_features[max_index][1]
        print('应该移除的特征值')
        print(remove_feature)
        print('共',len(remove_feature),'个')
        return list(remove_feature)


    '''
    调参数
    
    '''
    #这里写代码片
    def paramers_adjust(self):
        # 特征值处理
        self.Correlation_coefficient()
        # 准备训练数据和y值 X_train, y_train = ...
        ratio = 0.7
        x_train = self.data_x[:int(len(self.data_x) * ratio)]
        print(x_train.shape)
        x_test = self.data_x[int(len(self.data_x) * ratio):]
        y_train = self.data_y[:int(len(self.data_y) * ratio)]
        y_test = self.data_y[int(len(self.data_y) * ratio):]

        #初步定义分类器
        rfc = RandomForestClassifier(n_estimators=800, max_depth=11, random_state = 0)
        #clf.fit(x_train, y_train)
        #需要选择的参数名称一起后选值
        tuned_parameter = [{'max_features':[20,25,30,35,40,45,50,55,60]}]
         #神器出场,cv设置交叉验证
        clf = GridSearchCV(estimator=rfc, param_grid=tuned_parameter, cv=3, n_jobs=1)
        #拟合训练集
        clf.fit(x_train, y_train)
        print('Best parameters:')
        print(clf.best_params_)
        # 使用选择出的最好的参数在测试集上进行测试，并且评分
        score = clf.score(x_test, y_test)
        print(score)

#Filter().paramers_adjust()