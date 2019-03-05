#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: get_stock_info.py
@time: 2018/12/02
"""
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
import tushare as ts
import dataBase.Connection as conn
import datetime as datetime
import copy


class get_stock_info(object):
    # 获得连接
    host = '127.0.0.1'
    port = 3306
    user = 'root'
    password = '1'
    db = 'stock'
    '''
    pymysql 是 数据库驱动， 
    MySQLdb 只适合用python2
    '''
    # conn = create_engine('mysql+pymysql://root:1@127.0.0.1:3306/stock?charset=utf8mb4')
    conn = conn.mysql_operator(host, port, user, password, db)

    # 获得对应行业的股票代码
    def get_special_industry_stock(self, industry_name):
        sql = 'SELECT stock_code,stock_name from stock_industry where industry1_name="'+industry_name+'"'
        print(sql)
        result = self.conn.select(sql, 'stock_code', 'stock_name')
        if result == None or len(result) < 1:
            print('not stock')
            return []
        return result

    # 获得所有的行业股票的所有的信息
    def get_info(self, industry_name):
        stock_code = self.get_special_industry_stock(industry_name)
        if stock_code == None:
            print('no stock')
            return
        for one in stock_code:
            # while True:
            #     k = 1000;
            #     if k<1:
            #         break
            #     k-=1
            try:
                # YYYY-MM-DD
                df = ts.get_hist_data(one[0], start='2018-12-10')
                #print(df)
                for index, row in df.iterrows():
                    print(index)
                    date = index
                    open = row['open']
                    high = row['high']
                    close = row['close']
                    low = row['low']
                    volume = row['volume']
                    price_change = row['price_change']
                    p_change = row['p_change']
                    ma5 = row['ma5']
                    ma10 = row['ma10']
                    ma20 = row['ma20']
                    v_ma5 = row['v_ma5']
                    v_ma10 = row['v_ma10']
                    v_ma20 = row['v_ma20']
                    try :
                        turnover = row['turnover']
                        sql = "INSERT INTO stock_info(stock_code, stock_name, date, open, high, close, low, volume, price_change, p_change, ma5, ma10, ma20, v_ma5, v_ma10, v_ma20, turnover) " \
                              "VALUES ('%s', '%s', '%s', '%s', '%s', '%s','%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')" % \
                              (one[0], one[1], date, open, high, close, low, volume, price_change, p_change, ma5, ma10, ma20, v_ma5, v_ma10, v_ma20, turnover)
                    except Exception as e1:
                        print('turnover is not exit')
                        sql = "INSERT INTO stock_info(stock_code, stock_name, date, open, high, close, low, volume, price_change, p_change, ma5, ma10, ma20, v_ma5, v_ma10, v_ma20) " \
                              "VALUES ('%s', '%s', '%s', '%s', '%s', '%s','%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')" % \
                              (one[0], one[1], date, open, high, close, low, volume, price_change, p_change, ma5, ma10, ma20, v_ma5, v_ma10, v_ma20)

                    # sql = "INSERT INTO stock_industry (stock_code, stock_name, date, open, high, close, low, volume, price_change, p_change, ma5, ma10, ma20, v_ma5, v_ma10, v_ma20, turnover) " \
                    #       "VALUES ('" + one[0] + "', '" + one[1] + "', '" + date + "', '" + open + "', '" + high + "', '" + close + "', '" + low + "'" \
                    #         ", '" + volume + "', '" + price_change + "', '" + p_change + "', '" + ma5 + "', '" + ma10 + "', '" + ma20 + "'" \
                    #                 ", '" + v_ma5 + "', '" + v_ma10 + "', '" + v_ma20 + "', '" + turnover + "');"

                    print(sql)
                    self.conn.insert(sql+';')
            except Exception as e:
                print(stock_code, e)


    '''
    根据股票的代码获取数据
    '''
    def get_info_from_code(self, code, code_name):
        df = ts.get_hist_data(code)
        # print(df)
        select_sql = 'SELECT * from stock_info where stock_code="' + code + '" order by date asc;'
        exist_set = self.conn.select(select_sql, 'date')

        date_set = list(map(lambda x:datetime.datetime.strftime(x, "%Y-%m-%d"),
                        exist_set))


        print(date_set)
        for index, row in df.iterrows():
            date = index
            if date in date_set:
                print('exist')
                continue
            open = row['open']
            high = row['high']
            close = row['close']
            low = row['low']
            volume = row['volume']
            price_change = row['price_change']
            p_change = row['p_change']
            ma5 = row['ma5']
            ma10 = row['ma10']
            ma20 = row['ma20']
            v_ma5 = row['v_ma5']
            v_ma10 = row['v_ma10']
            v_ma20 = row['v_ma20']
            try:
                turnover = row['turnover']
                sql = "INSERT INTO stock_info(stock_code, stock_name, date, open, high, close, low, volume, price_change, p_change, ma5, ma10, ma20, v_ma5, v_ma10, v_ma20, turnover) " \
                      "VALUES ('%s', '%s', '%s', '%s', '%s', '%s','%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')" % \
                      (code, code_name, date, open, high, close, low, volume, price_change, p_change, ma5, ma10, ma20,
                       v_ma5, v_ma10, v_ma20, turnover)
            except Exception as e1:
                # print('turnover is not exit')
                sql = "INSERT INTO stock_info(stock_code, stock_name, date, open, high, close, low, volume, price_change, p_change, ma5, ma10, ma20, v_ma5, v_ma10, v_ma20) " \
                      "VALUES ('%s', '%s', '%s', '%s', '%s', '%s','%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')" % \
                      (code, code_name, date, open, high, close, low, volume, price_change, p_change, ma5, ma10, ma20,
                       v_ma5, v_ma10, v_ma20)

            # sql = "INSERT INTO stock_industry (stock_code, stock_name, date, open, high, close, low, volume, price_change, p_change, ma5, ma10, ma20, v_ma5, v_ma10, v_ma20, turnover) " \
            #       "VALUES ('" + one[0] + "', '" + one[1] + "', '" + date + "', '" + open + "', '" + high + "', '" + close + "', '" + low + "'" \
            #         ", '" + volume + "', '" + price_change + "', '" + p_change + "', '" + ma5 + "', '" + ma10 + "', '" + ma20 + "'" \
            #                 ", '" + v_ma5 + "', '" + v_ma10 + "', '" + v_ma20 + "', '" + turnover + "');"


            print(sql)
            print('插入一个')
            self.conn.insert(sql +  ';')
        del date_set
        del exist_set



gsi = get_stock_info()
gsi.get_info_from_code('300324', '旋极信息')
#stock_code = gsi.get_special_industry_stock('电子信息')


# for one in stock_code:
#     try:
#         gsi.get_info_from_code(one[0], one[1])
#     except Exception:
#         print('error', one)
#print(result)