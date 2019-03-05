#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: K_mul_lines.py
@time: 2018/11/26
"""
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#from STA_indicators import K_lines
import STA_indicators.K_lines as kl

# 所有的K线的多日指标
# 所有的k线的指标的基准是以开盘价为标准


class K_mul_lines(kl):

    def __init__(self):
        pass

    # 希望之星
    # 第一根是阴线，二是十字线，三是阳线
    def hope_star(self, opens, closes, highs, lowers):
        index = 1
        for open, close, high, lower in opens, closes, highs, lowers:
            if index == 1:
                if kl.K_lines.yin_line(open, close, high, lower):
                    index += 1
                    continue
            if index == 2 :
                if kl.K_lines.reticle(open, close, high, lower):
                    index += 1
                    continue
                else:
                    index = 1
            if index == 3:
                if kl.K_lines.sun_line(open, close, high, lower):
                    return True
                else:
                    index = 1
    # 希望之星
    # 第一根是阳线，二是十字线，三是阴线
    def evening_star(self, opens, closes, highs, lowers):
        index = 1
        for open, close, high, lower in opens, closes, highs, lowers:
            if index == 1:
                if kl.K_lines.sun_line(open, close, high, lower):
                    index += 1
                    continue
            if index == 2:
                if kl.K_lines.reticle(open, close, high, lower):
                    index += 1
                    continue
                else:
                    index = 1
            if index == 3:
                if kl.K_lines.yin_line(open, close, high, lower):
                    return True
                else:
                    index = 1

    #看涨反击线
    # 先是一根大阴线，后来是调低开盘 收出一根阳线 收于前一根K线的收盘价
    def dull_counter_line(self, opens, closes, highs, lowers):
        index = 1
        last_close = 0.0
        last_lower = 0.0
        for open, close, high, lower in opens, closes, highs, lowers:
            if index == 1:
                last_close = close
                last_lower = lower
                if kl.K_lines.yin_line(open, close, high, lower):
                    index += 1
                    continue
            if index == 2 and (open < last_lower):
                if kl.K_lines.sun_line(open, close, high, lower) and (abs(close-
                        last_close)<last_close*0.01):
                    return True

    #看跌反击线
    # 先是一根大阳线，后来是调高开盘 收出一根阴线 收于前一根K线的收盘价
    def dull_counter_line(self, opens, closes, highs, lowers):
        index = 1
        last_close = 0.0
        last_high = 0.0
        for open, close, high, lower in opens, closes, highs, lowers:
            if index == 1:
                last_close = close
                last_high = high
                if kl.K_lines.sun_line(open, close, high, lower):
                    index += 1
                    continue
            if index == 2 and (open > last_high):
                if kl.K_lines.yin_line(open, close, high, lower) and (abs(close-
                        last_close)<last_close*0.01):
                    return True
    #曙光初现
    # 一根阴线，接着一根阳线,阳线的实体插入阴线的1/2处
    def dawn_breaks(self, opens, closes, highs, lowers):
        index = 1
        last_close = 0.0
        last_open = 0.0
        for open, close, high, lower in opens, closes, highs, lowers:
            if index == 1:
                last_close = close
                last_open = open
                if kl.K_lines.yin_line(open, close, high, lower):
                    index += 1
                    continue
            if index == 2:
                if kl.K_lines.sun_line(open, close, high, lower) and close >= (last_close+last_open)*0.4999:
                    return True

    #乌云密布
    # 一根阳线，接着一根阴线,阴线的实体插入阳线的1/2处
    def dark_cloud(self, opens, closes, highs, lowers):
        index = 1
        last_close = 0.0
        last_open = 0.0
        for open, close, high, lower in opens, closes, highs, lowers:
            if index == 1:
                last_close = close
                last_open = open
                if kl.K_lines.sun_line(open, close, high, lower):
                    index += 1
                    continue
            if index == 2 :
                if kl.K_lines.yin_line(open, close, high, lower) and close <= (last_close + last_open)*0.5001:
                    return True