#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: unique_csv.py
@time: 2018/12/05
"""
import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd


def unique_file(old_file_path, new_file_path, key_index, key_word=None):
    '''

    :param fill_path:
    :param key_index: 去重的关键字段下标（主要是针对可以拆分的文件）
    :param key_word: 去重的关键字段

    :return:
    '''
    data_set = set()

    if key_word == None:
        pass

    if key_index != None:
        try:
            new_file = open(new_file_path, 'w+')
            with open(old_file_path) as f:
                while True:
                    line = f.readline()
                    if line:
                        words = line.split(',')
                        key = words[key_index]
                        if data_set.__contains__(key):
                            print('包含'+key)
                        else:
                            data_set.add(key)
                            new_file.write(line)
                            new_file.flush()
                    else:
                        break
            new_file.close()
        except Exception as e:
            print('error>>>', e)


def classfic_4(data):
    one = data
    if one <= -5:
        one = -2
    elif 0> one > -5:
        one = -1
    elif one == 0 :
        one = 0
    elif 5 > one > 0:
        one = 1
    else:
        one = 2
    return one


data = [1,2,3,-5,-3.4,6,0,0]
#print(list(map(classfic_4, data)))