#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: Essemble_main.py
@time: 2019/01/17
"""
from STA_indicators.STA_main import get_other_indicators
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from algoruthm.unsupervised_Learning import Mean_shift_mars as ms
import Feature_Engineering.Normalizer as nr
from algoruthm.dimensionality_reduction import PCA_mars
import Feature_Engineering.outliner_check as oc
from algoruthm.supervision_learning.randomForest import random_forest,getScore
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import datasets, svm
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop



def dataX_from_dataFrame(data=pd.DataFrame({})):
    temp_x = []
    for indexs in data.index:
        temp_x.append(data.loc[indexs])
        #print(data.loc[indexs].values)
    # X在此不取最后一个值，为了用ｔ的值预测ｔ＋１的值
    x = np.copy(temp_x[:-1])
    return x

# 二分类对Ｙ值做处理
def dataY_from_dataFrame(data=pd.DataFrame({})):
    # 以price_change做相应的二值化处理，作为Ｙ值
    # price_change大于０的为１，小于０的为０。自此作为股票的涨跌
    y = data.get('price_change')[1:]
    y = y.iloc[:, 0]
    #print('price_change', y)
    y = np.where(y > 0, 1, 0)
    return y



stock_code = '\'000032'

# 获得数据
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
    #print('code:', code[1:])
    path = '/home/mars/Data/finialData/electronic_infomation/' + code[1:] + '.csv'
    code_data = pd.read_csv(path, index_col='date')

    result = get_other_indicators(code_data)
    # 数据整合
    dataList.append(result)
# 按照时间对接，并且去掉NAN数据
df = pd.concat(dataList, axis=1, sort=False)

# pandas会 按照文件的index索引来进行重新的拼接
new_df = df.sort_index()
#print('new_df:', new_df[:5])

new_df.dropna(axis=0, inplace=True)
#print('new_df2:', new_df.get('price_change'))
#print('all shape:', new_df.shape)
deal_result = new_df

data_x = dataX_from_dataFrame(deal_result)
data_y = dataY_from_dataFrame(deal_result)
print('data_y', data_y.shape)

# data_y = final_data[1]
final_data_x = nr.standardized_mars(data_x)

y_real = data_y[int(len(data_y) * 0.7):]


# 获得MS的train_Y
print('***********开始测试 MS ********************')
MSx_train, MSx_test, MSy_train, MSy_test = ms.getMS_repx_data(final_data_x, data_y)
print('MSy_train,MSy_test:', MSy_train.shape,MSy_test.shape )
all_x = np.vstack((MSx_train, MSx_test))
all_y = np.concatenate((MSy_train, MSy_test), axis=0)
pca_x = PCA_mars.getPcaComponent(all_x, n_components=35)
print('all_y:',all_y.shape)
MS_predict_y = random_forest((pca_x, all_y))
print('MS_predict_y',MS_predict_y.shape)
ms_score = round(getScore(), 4)
del pca_x
del all_y
del all_x

# 获得singular的train_Y
print('***********开始测试 singular ********************')
pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=50)
lof_data_x = oc.LOF_PCA_for_Clustering(pca_x, isUsePCA=False)

singular_predict_y = random_forest((lof_data_x, data_y))
print('singular_predict_y',singular_predict_y.shape)
singular_score = round(getScore(),4)
del lof_data_x
del pca_x

# 获得original_RF的train_Y
print('***********开始测试 original ********************')
pca_x = PCA_mars.getPcaComponent(final_data_x, n_components=53)
original_predict_y = random_forest((pca_x, data_y))
print('original_predict_y',original_predict_y.shape)
original_score = round(getScore(),4)
del pca_x
# 获得model4的train_Y

# 组合所有的train_y将其转换成我们新的data_x
#print(MS_predict_y.reshape(-1,1))
#print(MS_predict_y.T[:10],'--', singular_predict_y[:10], '--', original_predict_y[:10])
essemble_x = np.hstack((MS_predict_y.reshape(-1,1), singular_predict_y.reshape(-1,1), original_predict_y.reshape(-1,1)))
eX_train, eX_test, ey_train, ey_test = train_test_split(essemble_x, y_real, random_state=0, train_size=0.6, shuffle=False)
# 获得真实的训练集Y

# fit model

model = Sequential()
'''
   第二步：构建网络层
'''
print(eX_train.shape)
model.add(Dense(50, input_shape=(3,)))  # 输入层，28*28=784
model.add(Activation('relu'))  # 激活函数是tanh
model.add(Dropout(0.1))  # 采用50%的dropout
model.add(Dense(1))  # 输出结果是10个类别，所以维度是10
model.add(Activation('softmax'))  # 最后一层用softmax作为激活函数

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  # 优化函数，设定学习率（lr）等参数
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode='categorical')  # 使用交叉熵作为loss函数

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(eX_train, ey_train, epochs=100, batch_size=5, shuffle=False, verbose=0, validation_split=0.0)
#model.evaluate(X_test, Y_test, batch_size=200, verbose=0)

'''
    第五步：输出
'''
print("test set")
loss, accuracy = model.evaluate(eX_test, ey_test, verbose=0)
print('loss:', loss)
print('accuracy:', accuracy)
print("")


'''
scoreListInfo = []
#for fig_num, kernel in enumerate(('poly','sigmoid')):
for c in np.arange(1, 10, 1):
    for gamma in np.arange(1, 10, 1):
        clf = svm.SVC(C=c, gamma=gamma, kernel='poly')
        clf.fit(eX_train, ey_train)
        score = clf.score(eX_test, ey_test)
        print(score)
        scoreListInfo.append((score, gamma, c))

    #print(scoreListInfo)
scoreList = []
for one in scoreListInfo:
    score = one[0]
    scoreList.append(score)
max_score = max(scoreList)
max_index = scoreList.index(max_score)
# error_ratio = scoreInfoList[max_index][1]
gamma = scoreListInfo[max_index][1]
c = scoreListInfo[max_index][2]
del scoreListInfo
del scoreList
print('best paramers:')
print(max_score, c, gamma)
'''

'''
linear_regression = LinearRegression().fit(eX_train, ey_train)
print(linear_regression.coef_)
print(linear_regression.intercept_)

# 为训练集打分
print("R^2 on training set: %f" % linear_regression.score(eX_train, ey_train))
# 将该模型用到测试集，为测试集打分
print("R^2 on test set: %f" % linear_regression.score(eX_test, ey_test))
# 对产出的 make_regression 性能评估
print(r2_score(np.dot(essemble_x, linear_regression.coef_), y_real))
# 对回归器的性能评估
print(r2_score(linear_regression.predict(essemble_x), y_real))
'''
# predict


#plot
'''
ms_predicts = []
s_predicts = []
o_predicts = []
new_test = []
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#ax.scatter(range(len(y_test)), y_test, label='real', c='red')
for (ms,s,o,t) in zip(list(MS_predict_y-0.02), list(singular_predict_y-0.04), list(original_predict_y-0.06), y_real):
    ms_predicts.append(ms)
    s_predicts.append(s)
    o_predicts.append(o)
    new_test.append(t)
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #anim = animation.FuncAnimation(fig, lambda i: scatter.set_data(new_predicts), interval=30, blit=True)
    ax.cla()
    ax.scatter(range(len(new_test)), new_test, label='real', c='red')
    ax.scatter(range(len(ms_predicts)), ms_predicts, label='ms_predict['+str(ms_score)+']', c='g')
    ax.scatter(range(len(s_predicts)), s_predicts, label='s_predict['+str(singular_score)+']', c='y')
    ax.scatter(range(len(o_predicts)), o_predicts, label='o_predict['+str(original_score)+']', c='b')
    ax.legend()
    plt.pause(0.6)
    plt.tight_layout()
plt.show()
'''