#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: randomForest.py
@time: 2018/12/07
"""
# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from stock_former.get_mian_indictors import *
from stockstats import *
from STA_indicators.STA_main import get_other_indicators
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os, sys
from tpot import TPOTClassifier
from tool.unique_csv import classfic_4
from matplotlib import animation

# 二分类
def deal_data_from_dataFrame(data=pd.DataFrame({})):
    y = data.get('price_change')[1:]
    y = np.where(y > 0, 1, 0)
    temp_x = []
    for indexs in data.index:
        temp_x.append(data.loc[indexs])
        #print(data.loc[indexs].values)

    x = np.copy(temp_x[:-1])
    return (x, y)

# 二分类对ｘ值做处理
def dataX_from_dataFrame(data=pd.DataFrame({})):
    temp_x = []
    for indexs in data.index:
        temp_x.append(data.loc[indexs])
        #print(data.loc[indexs].values)
    # X在此不取最后一个值，为了用ｔ的值预测ｔ＋１的值
    x = np.copy(temp_x[:-1])
    return x

# 二分类对Ｙ值处理有相关系数的股票做处理
def dataY_from_dataFrame(data=pd.DataFrame({})):
    # 以price_change做相应的二值化处理，作为Ｙ值
    # price_change大于０的为１，小于０的为０。自此作为股票的涨跌
    y = data.get('change')[1:]
    y = y.iloc[:, 0]
    #print('price_change', y)
    y = np.where(y > 0, 1, 0)
    return y


# 二分类对Ｙ值做处理
def dataY_no_correlation(data=pd.DataFrame({})):
    # 以price_change做相应的二值化处理，作为Ｙ值
    # price_change大于０的为１，小于０的为０。自此作为股票的涨跌
    y = data.get('change')[1:]
    #print('price_change', y)
    y = np.where(y > 0, 1, 0)
    return y


# 4分类对Ｙ值做处理
def dataY_from_dataFrame_5(data=pd.DataFrame({})):
    # 以price_change做相应的二值化处理，作为Ｙ值
    # price_change大于０的为１，小于０的为０。自此作为股票的涨跌
    y = data.get('change')[1:]
    y = y.iloc[:, 0]
    #print('price_change', y)
    y = np.copy(list(map(classfic_4, y)))
    #print(y)
    return y

# 4分类对Ｙ值做处理没有相关股票
def dataY_5_no_correlation(data=pd.DataFrame({})):
    # 以price_change做相应的二值化处理，作为Ｙ值
    # price_change大于０的为１，小于０的为０。自此作为股票的涨跌
    y = data.get('change')[1:]
    #print('price_change', y)
    y = np.copy(list(map(classfic_4, y)))
    return y




# 二分类,将所有的指标取N天的平均值为一个数据，所以总的数据量会减少(N-1)/N
def deal_data_for_Nmean(data=pd.DataFrame({}), N=3):
    '''
    :param data:
    :param N: 取值的单位
    :return:
    '''
    temp_y = data.get('price_change')
    y = get_all_meanN(temp_y, N=N)[1:]
    y = np.where(y > 0, 1, 0)
    print('y的维数',y.shape)
    temp_x = []
    for indexs in data.index:
        temp_x.append(data.loc[indexs])
    x = np.copy(get_all_meanN(temp_x, N=N)[:-1])
    print('x的维数',x.shape)
    return (x, y)

# 二分类时间跨度对ｘ值做处理
def dataX_for_Nmean_no(data=pd.DataFrame({}), N=3):
    temp_x = []
    for indexs in data.index:
        temp_x.append(data.loc[indexs])
    x = np.copy(get_all_meanN(temp_x, N=N)[:-1])
    print('x的维数', x.shape)
    return x

# 二分类时间跨度对Ｙ值做处理
def dataY_for_Nmean(data=pd.DataFrame({}), N=3):
    temp_y = data.get('price_change')[1:]
    y = get_all_meanN(temp_y, N=N)
    y = np.where(y > 0, 1, 0)
    print('y的维数', y.shape)
    return y


# 多分类
def deal_data_for_multiclassfy(data=pd.DataFrame({})):
    y = data.get('p_change')[1:]
    #y = np.where(y > 0, 1, 0)
    y = np.copy([int(one) for one in y])
    temp_x = []
    for indexs in data.index:
        temp_x.append(data.loc[indexs])
        #print(data.loc[indexs].values)

    x = np.copy(temp_x[:-1])


    return (x, y)



def random_forest_up1(data, ratio=0.7, classes=[]):
    '''

    :param data:tuple类型
    :param ratio: 训练集的占有率
    :param classes: 分类的种类
    :return:
    '''
    data_x = data[0]
    data_y = data[1]
    x_train = data_x[:int(len(data_x)*ratio)]
    print('x_train',x_train.shape)
    x_test = data_x[int(len(data_x)*ratio):]
    y_train = data_y[:int(len(data_y)*ratio)]
    y_test = data_y[int(len(data_y) * ratio):]
    #print('y_test', y_test)
    # 随机森林模型
    clf = RandomForestClassifier(n_estimators=1000, max_depth=20, random_state = 0)
    clf.fit(x_train, y_train)
    print('特征的贡献比利：')
    #print(clf.feature_importances_)
    score = clf.score(x_test, y_test)
    global test_score
    test_score = score
    #print('测试的结果：')
    #print(clf.predict(x_test))
    print('模型的分数：', score)



#ClassifierMixin
def random_forest(data, ratio=0.7, data_index=[], another_data_x=[]):
    '''

    :param data:tuple类型
    :param ratio: 训练集的占有率
    :param classes: 分类的种类
    :return:
    '''
    print('another_data_x:', another_data_x)
    data_x = data[0]
    data_y = data[1]
    x_train = data_x[:int(len(data_x)*ratio)]
    print('x_train',x_train.shape)
    x_test = data_x[int(len(data_x)*ratio):]
    y_train = data_y[:int(len(data_y)*ratio)]
    y_test = data_y[int(len(data_y) * ratio):]

    #test_date = data_index[int(len(data_y) * ratio):]

    #print('y_test', y_test)
    # 随机森林模型
    clf = RandomForestClassifier(n_estimators=1000, max_depth=80, random_state = 0)
    clf.fit(x_train, y_train)
    print('特征的贡献比利：')
    #print(clf.feature_importances_)
    score = clf.score(x_test, y_test)
    global test_score
    test_score = score
    #print('测试的结果：')
    #print(clf.predict(x_test))
    print('模型的分数：', score)

    # Compute confusion matrix
    '''
    cnf_matrix = confusion_matrix(y_test, clf.predict(x_test))
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix,classes=classes,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix')

    plt.show()
    plt.close()
    plt.scatter(range(len(y_test)), y_test, label='real', s=40)
    plt.scatter(range(len(x_test)), clf.predict(x_test), label='test', s=10)
    plt.legend()
    plt.tight_layout()
    plt.show()
    '''
    '''
    predicts = clf.predict(x_test)
    predicts = predicts - 0.1
    new_predicts = []
    new_test = []
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax.scatter(range(len(y_test)), y_test, label='real', c='red')
    for (i,j) in zip(list(predicts), y_test):
        new_predicts.append(i)
        new_test.append(j)
        #fig, (ax1, ax2) = plt.subplots(1, 2)
        #anim = animation.FuncAnimation(fig, lambda i: scatter.set_data(new_predicts), interval=30, blit=True)
        ax.cla()
        ax.scatter(range(len(new_test)), new_test, label='real', c='red')
        ax.scatter(range(len(new_predicts)), new_predicts, label='predict', c='green')
        #ax.legend()
        plt.pause(0.6)
        #plt.tight_layout()
    plt.show()
    '''


    '''
    实际值与预测值比较
    predicts = clf.predict(x_test)
    print(predicts)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(range(len(y_test)), y_test, label='real', c='green', s=3)
    ax2.scatter(range(len(x_test)), predicts, label='predict', c='blue', s=3)
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()
    '''
    if len(another_data_x)>0:
        return (clf.predict(x_test),  clf.predict(another_data_x))
    return clf.predict(x_test)


#Regression
def random_forest_for_Regression(data, ratio=0.7, trees=100):
    # 标准化
    data_x = StandardScaler().fit_transform(data[0])
    data_y = data[1]
    x_train = data_x[:int(len(data_x)*ratio)]
    print(x_train.shape)
    x_test = data_x[int(len(data_x)*ratio):]
    y_train = data_y[:int(len(data_y)*ratio)]
    y_test = data_y[int(len(data_y) * ratio):]
    print(y_test.shape)
    clf = RandomForestRegressor(n_estimators=trees, max_depth=10, random_state = 0)
    clf.fit(x_train, y_train)
    '''
    0 表示模型效果跟瞎猜差不多
    1 表示模型拟合度较好（有可能会是过拟合，需要判定）
    0~1 表示模型的好坏（针对同一批数据）
    小于0则说明模型效果还不如瞎猜（说明数据直接就不存在线性关系）
    '''
    score = clf.score(x_test, y_test)
    print(score)
    #print(clf.feature_importances_)
    # 计算测试集上的预测值
    prediction = clf.predict(x_test)
    ##计算均方差并加入到列表
    error = mean_squared_error(y_test, prediction)
    plt.plot(prediction, label='predict')
    plt.plot(y_test, label='real')
    plt.legend()
    plt.show()
    return error






'''
获得模型在测试集上的分数
'''
def getScore():
    return test_score



"""
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
def plot_confusion_matrix(cm, classes=[],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    '''

    :param cm:
    :param classes: 分类的ｌｉｓｔ
    :param normalize:
    :param title:
    :param cmap:
    :return:
    '''

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()





'''
基于遗传算法的模型选择
'''
def choose_model():
    #使用遗传算法选择模型
    data = pd.read_csv('/home/mars/Data/000032.csv')
    # 按时间的增序排列
    data = data[::-1]
    # 加入其他的指标
    result = get_other_indicators(data)
    # 除掉数据为ＮＡＮ的行数据
    deal_result = result.dropna(axis=0)
    # 对Ｙ值做二分化处理
    s_deal_result = deal_data_from_dataFrame(deal_result)

    # 对数据做不同的时间跨度的处理,包括Ｙ值的处理
    # s_deal_result = deal_data_for_Nmean(deal_result, N=3)

    # 特征选择，　也可以不做特征选择
    #final_data = Filter(use=False).Variance_selection(threshold=3, data=s_deal_result)
    final_data = []
    data_x = final_data[0]
    data_y = final_data[1]
    ratio = 0.7
    x_train = data_x[:int(len(data_x) * ratio)]
    print(x_train.shape)
    x_test = data_x[int(len(data_x) * ratio):]
    y_train = data_y[:int(len(data_y) * ratio)]
    y_test = data_y[int(len(data_y) * ratio):]
    '''
    max_time_mins：最大的测试时间（分钟为单位）
    mutation_rate：　变异概率
    crossover_rate：　交换概率
    n_jobs：　线程数
    generations: 遗传算法进化次数，可理解为迭代次数
    population_size: 每次进化中种群大小
    '''
    tpot = TPOTClassifier(verbosity=2, max_time_mins=40, config_dict="TPOT light", population_size=50, mutation_rate=0.9,
                          crossover_rate=0.1, n_jobs=-1)

    tpot.fit(x_train.astype(float), y_train.astype(float))
    # 利用测试的最优的算法进行在测试集上的测试
    print(tpot.score(x_test.astype(float), y_test.astype(float)))



#random_forest(treat_data_for_RF())

#treat_data_for_RF()




'''
测试一只股票的模型
'''
def main():
    # 记录每轮的分数和对应删除特征值
    score_features = []

    data = pd.read_csv('/home/mars/Data/000032.csv')
    # 按时间的增序排列
    data = data[::-1]
    # 加入其他的指标
    result = get_other_indicators(data)
    #result = pd.DataFrame(data)
    # 除掉数据为ＮＡＮ的行数据
    deal_result = result.dropna(axis=0)
    # 对X处理, Ｙ值做二分化处理
    data_x = dataX_from_dataFrame(deal_result)
    data_y = dataY_from_dataFrame(deal_result)
    s_deal_result = (data_x, data_y)
    # 对数据做不同的时间跨度的处理,包括Ｙ值的处理
    #s_deal_result = deal_data_for_Nmean(deal_result, N=3)

    # 特征选择，　也可以不做特征选择
    #final_data = Filter(use=False).Variance_selection(threshold=3, data=s_deal_result)
    feature_importances = np.copy(random_forest(s_deal_result))
    delete_feature = []
    # 初始的特征值不用删除，

    score_features.append((getScore(),np.copy(delete_feature)))

    columns = list(deal_result.columns.values)
    #print(feature_importances)
    number = len(feature_importances)
    # 除掉特征贡献小于　1 / number　的特征
    for i in range(number):
        if feature_importances[i] < (1 / number):
            # delect_index.append(i)
            delete_feature.append(columns[i])
    # if 'price_change' in delete_feature:
    #     delete_feature.remove('price_change')

    # # 获得需要删除的列名
    # for i in delect_index:
    #     delete_feature.append(columns[i])

    # 去掉贡献很小的features
    for round in range(1, 10):
        # 被删除的特征值不能少于原数量的一半
        if len(delete_feature) <= int(number*2/3):
            if delete_feature == []:
                break
            else:
                # 去掉贡献值小的特征值
                deal_result_2 = deal_result.drop(labels=delete_feature, axis=1)
                print(delete_feature)
                print(round, '>>***')
                columns_2 = list(deal_result_2.columns.values)
                print(len(columns_2))
                final_data_2_X = dataX_from_dataFrame(deal_result_2)
                final_data_2 = (final_data_2_X, data_y)
                feature_importances_2 = random_forest(final_data_2)
                score_features.append((getScore(), np.copy(delete_feature)))
                #print(feature_importances_2)
                # 讲贡献值最小的两位排除
                for i in range(2):
                    min_index = list(feature_importances_2).index(min(feature_importances_2))
                    delete_feature.append(columns_2[min_index])
                del(deal_result_2)
                del(columns_2)
                del(final_data_2_X)
                del(feature_importances_2)
        else:
            break
    # 比较得出最高的分数，并且输出对应的特征值
    scoreList = []
    for one in score_features:
        score = one[0]
        scoreList.append(score)

    max_socre = max(scoreList)
    max_index = scoreList.index(max_socre)
    remove_feature = score_features[max_index][1]
    print(score_features)
    print(max_socre, remove_feature)
'''
测试多分类
'''

def main_mul_class():
    data = pd.read_csv('/home/mars/Data/000032.csv')
    # 按时间的增序排列
    data = data[::-1]
    # 加入其他的指标
    result = get_other_indicators(data)
    delete_feature = []
    deal_result = result.dropna(axis=0)
    # print(deal_result)
    print('***')
    columns = list(deal_result.columns.values)
    #print(len(columns))
    final_data = deal_data_for_multiclassfy(deal_result)
    classes = range(-10,11,1)
    feature_importances = random_forest(final_data, classes=classes)
    print(feature_importances)





'''
测试多个股票
'''
def main_3():
    # 股票存放的集合
    path = '/home/mars/Data/finialData/electronic_infomation/'
    parents = os.listdir(path)
    # 存放不同的股票的测试分数
    scoreList = []
    for parent in parents:
        child = os.path.join(path, parent)
        m_data = pd.read_csv(child)
        m_data = m_data[::-1]
        result = get_other_indicators(m_data)
        deal_data = result.dropna(axis=0)
        # 对数据做不同的时间跨度的处理
        deal_result = deal_data_for_Nmean(deal_data, N=3)
        #特征选择
        #final_data = Filter(use=False).Variance_selection(threshold=3, data=deal_result)
        final_data = []
        # print(deal_result)
        print('***')
        # print(len(columns))
        random_forest(final_data, ratio=0.7)
        scoreList.append((getScore(), parent))

    print(scoreList)


def test(daySpan=0):
    data = pd.read_csv('/home/mars/Data/000032.csv')
    # 按时间的增序排列
    data = data[::-1]
    # 记录每轮的分数和对应删除特征值

    # 按时间的增序排列
    data = data[::-1]
    # 加入其他的指标
    result = get_other_indicators(data)
    # result = pd.DataFrame(data)
    # 除掉数据为ＮＡＮ的行数据
    deal_result = result.dropna(axis=0)
    data_x = dataX_from_dataFrame(deal_result)
    if daySpan == 0:
        # 对X处理, Ｙ值做二分化处理

        data_y = dataY_from_dataFrame(deal_result)
    else:
        data_y = dataY_for_Nmean(deal_result, N=2)
    s_deal_data = (data_x, data_y)
    # 特征选择
    #delete_feature = Filter(use=False).feature_RandomForest(deal_result=deal_result, final_data=s_deal_data, data_y=data_y, cicle=3)
    delete_feature = []
    # 除去多余的特征
    t_deal_result = deal_result.drop(labels=delete_feature, axis=1)
    final_data_X = dataX_from_dataFrame(t_deal_result)
    final_data = (final_data_X, data_y)
    # ｆｉｔ模型
    print('最佳特征值测试模型：')
    print('')
    random_forest(final_data)
