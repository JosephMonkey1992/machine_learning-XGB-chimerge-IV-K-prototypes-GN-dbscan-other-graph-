import pandas as pd
import datetime
# import jieba
import re
from collections import Counter
import matplotlib.pyplot as plt
import sys
import warnings
import time
import math
from dateutil.relativedelta import relativedelta
import matplotlib
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from pyforest import *

warnings.filterwarnings('ignore')

class Calc_IV_WOE:
    def __init__(self, data, map_list,max_section_num=None,attr_list=None,label_name=None,data_type=None):
        self.mp = map_list
        self.d = data
        self.attr = attr_list
        self.label = label_name
        self.max_section_num = max_section_num
        self.type = data_type

    def calc_woe(self,z=None):
        temp = self.d[[self.attr[z],self.label]]
        temp['stas'] = 1
        X = temp[temp[self.label]==1]['stas'].sum()
        Y = temp[temp[self.label]==0]['stas'].sum()
        iv_list = []
        if self.type == 'continue':
            attr = self.attr[z]
            s = self.mp[self.mp['属性名称']==attr]
            s.reset_index(inplace=True,drop=True)
            for i in range(self.max_section_num):
                a = float(s.iloc[i]['范围'].split('~')[0])
                b = float(s.iloc[i]['范围'].split('~')[1])
                if z!=self.max_section_num-1:
                    if a!=b:
                        temp1 = temp[(temp[self.attr[z]]>=a)&(temp[self.attr[z]]<b)]
                    else:
                        temp1 = temp[(temp[self.attr[z]]==a)]
                else:
                    temp1 = temp[(temp[self.attr[z]]>=a)&(temp[self.attr[z]]<=b)]
                x = temp1[temp1[self.label]==1]['stas'].sum()
                y = temp1[temp1[self.label]==0]['stas'].sum()
                woe = math.log((x/y)/(X/Y))
                vertical = (x/X)-(y/Y)
                iv = vertical*woe
                iv_list.append(iv)
        elif self.type == 'discrete':
            attr = self.attr[z]
            s = self.mp[self.mp['属性名称']==attr]
            s.reset_index(inplace=True,drop=True)
            max_num = len(s.index.values)
            for i in range(max_num):
                a = float(s.iloc[i]['范围'].split('~')[0])
                b = float(s.iloc[i]['范围'].split('~')[1])
                if z!=self.max_section_num-1:
                    if a!=b:
                        temp1 = temp[(temp[self.attr[z]]>=a)&(temp[self.attr[z]]<b)]
                    else:
                        temp1 = temp[(temp[self.attr[z]]==a)]
                else:
                    temp1 = temp[(temp[self.attr[z]]>=a)&(temp[self.attr[z]]<=b)]
                x = temp1[temp1[self.label]==1]['stas'].sum()
                y = temp1[temp1[self.label]==0]['stas'].sum()
                woe = math.log((x/y)/(X/Y))
                vertical = (x/X)-(y/Y)
                iv = vertical*woe
                iv_list.append(iv)
        else:
            pass
        iv_ = sum(iv_list)
        return iv_

    def calc_iv(self):
        final = pd.DataFrame(columns=['属性名称','information_value_IV'])
#         temp = pd.DataFrame(columns=['属性名称','information_value(IV)'])
        raw1= []
        raw2 = []
        for i in range(len(self.attr)):
#             raw = pd.DataFrame(columns=['属性名称','information_value_IV'])
            iv = self.calc_woe(z=i)
#             print(iv)
#             raw['属性名称'] = self.attr[i]
#             raw['information_value_IV'] = iv
            raw1.append(self.attr[i])
            raw2.append(iv)
#             print(self.attr[i])
#             print(iv)
        final['属性名称'] = raw1
        final['information_value_IV'] = raw2
        final = final.sort_values(by='information_value_IV',ascending=False)
        final.reset_index(inplace=True,drop=True)
        return final


if __name__ == '__main__':
    # here test_continue 是一个pandas sataframe....
    calc = Calc_IV_WOE(test_continue, map_frame,max_section_num=10,attr_list=list(test_continue.columns.values[0:-1]),label_name='if_convert',data_type='continue')
    iv = calc.calc_iv()

print('数据离散化结果：')
map_frame

print('各特征计算后IV值：')
iv