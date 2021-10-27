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


"""
计算置信区间方法（已知样本量，总体，重要性水平，样本准确率（近似均值））
"""
import json
import math


class intervalcalc:
    """
    para1: 计算置信区间
    para2: 判断样本量是否满足中心极限定律，是否满足大样本假设
    para3: 将有限总体修正系数加入置信区间计算（wald interval），其中1/2n是Wald Interval的连续性修正,使其更加趋向于二项分布的特性。
    """

    def __init__(self, n=None, N=None, alpha=None, rate=None):
        # 样本量
        self.n = n
        # 总体
        self.s = N
        # 重要性水平选择
        self.a = alpha
        # 样本准确率（期望）
        self.r = rate

    def calc(self):
        z = Z_score_map[str(self.a)]
        sigma = math.pow(self.r * (1 - self.r) / (self.n - 1), 0.5)
        f = math.pow(1 - (self.n / self.s), 0.5)
        interval = z * f * sigma + (1 / (2 * self.n))
        uplevel = self.r + interval
        lowlevel = self.r - interval
        output = '置信度:±' + str(round(interval * 100, 2)) + '%/  置信区间为:[' + str(round(uplevel * 100, 2)) + '%,' + str(
            round(lowlevel * 100, 2)) + '%]'
        return output

    def output(self):
        x = self.r * self.n
        y = self.n * (1 - self.r)
        if x > 5 and y > 5 and self.n >= 30:
            return self.calc()
        else:
            return '样本量不足!'

    def get(self):
        if self.a not in ['0.05', '0.01', '0.1', '0.02']:
            return '请选择重要性水平（浮点数）：0.1，0.05，0.02，0.01！'
        elif isinstance(self.r, float) and self.r <= float(1):
            return self.output()
        else:
            return '请输入正确格式样本准确率（浮点数）！该比率小于等于1.'


"""
Input
"""
if __name__ == '__main__':
    Z_score_map = {'0.05': 1.96, '0.01': 2.58, '0.1': 1.64, '0.02': 2.33}  # 由于是two-side检验，因此为Z(α/2)
    print('输入总体数量（整数）:')
    population_size = int(input())
    print('输入样本量（整数）:')
    sample_size = int(input())
    print('输入重要性水平（浮点数）:')
    significant_level = str(input())
    print('输入样本准确率(浮点数):')
    sample_rate = float(input())

"""
output
"""
print('=====================================')
print(intervalcalc(n=sample_size, N=population_size, alpha=significant_level, rate=sample_rate).get())

"""
计算样本量
"""
import json
import math


class samplecalc:
    """
    para1: 根据比率区间样本容量估计公式：n = Z(α/2)^2 * π(1-π) / E^2
    para2: 前提条件满足大样本假设及n(p)>5;n(1-p)>5;n>=30。(并未考虑ss<5%population有限总体修正条件的设置)
    para3: 考虑到有限总体修正因素，我们的的我们的样本量公式修正
    """

    def __init__(self, E=None, N=None, pai=None, alpha=None, size=None):
        # 误差估计
        self.e = E
        # 总体
        self.n = N
        # 重要性水平选择
        self.a = alpha
        # 总体比率估计
        self.p = pai
        # 判断是否有总体比率的初步估计
        self.s = size

    def calc(self):
        z = Z_score_map[str(self.a)]
        significance = math.pow(z, 2)
        variance1 = self.p * (1 - self.p)
        variance2 = 0.5 * (1 - 0.5)
        error = math.pow(self.e, 2)
        n1 = significance * variance1 / error
        n2 = significance * variance2 / error
        n1 = n1 / (1 + ((n1 - 1) / float(self.n)))
        n2 = n2 / (1 + ((n2 - 1) / float(self.n)))
        if self.s == '无':
            output = '样本量估计：无---样本量参考：≥' + str(n2)
        elif self.s == '有':
            output = '最小样本量：≥' + str(n1) + '---最佳样本量：≥' + str(n2)
        else:
            output = '请填写正确的总体比率初步估计：有/无。'
        return output, n1, n2

    def output(self):
        n1 = self.calc()[1]
        n2 = self.calc()[2]
        x1 = n1 * self.p
        y1 = n1 * (1 - self.p)
        x2 = n2 * 0.5
        y2 = n2 * (1 - 0.5)
        if n1 != 0:
            if x1 > 5 and y1 > 5 and n1 >= 30:
                return self.calc()[0]
            elif x2 > 5 and y2 > 5 and n2 >= 30:
                return self.calc()[0]
            else:
                return '实时样本量不足!'
        elif n1 == 0:
            if x2 > 5 and y2 > 5 and n2 >= 30:
                return self.calc()[0]
            else:
                return '实时样本量不足！'

    def get(self):
        if self.a not in ['0.05', '0.01', '0.1', '0.02']:
            return '请选择重要性水平（浮点数）：0.1，0.05，0.02，0.01！'
        elif isinstance(self.e, float) and self.e <= 0.1:
            return self.output()
        else:
            return '请输入正确格式的误差允许值！该比率小于等于0.1.'


"""
Input
"""
if __name__ == '__main__':
    Z_score_map = {'0.05': 1.96, '0.01': 2.58, '0.1': 1.64, '0.02': 2.33}  # 由于是two-side检验，因此为Z(α/2)
    print('输入总体数量（整数）:')
    population_size = int(input())
    print('输入重要性水平（浮点数）:')
    significant_level = str(input())
    print('是否有总体比率估计（*预计实时总体审出量准确率估计）:有/无')
    size = str(input())
    if size == '有':
        print('如果有总体比率估计，请填写估计比率（浮点数）')
        pai = float(input())
    elif size == '无':
        pai = float(0)
    else:
        pai = float(0)
    print('请填写误差允许范围：如0.05/0.04/0.1/0.02/0.01')
    E = float(input())

"""
output
"""
print('=====================================')
print(samplecalc(E=E, N=population_size, pai=pai, alpha=significant_level, size=size).get())



"""
word cloud
"""
from wordcloud import WordCloud
# fontpath='SourceHanSansCN-Regular.otf'
wc = WordCloud(font_path="simsun.ttf",  # 设置字体
               background_color="white",  # 背景颜色
               max_words=1000,  # 词云显示的最大词数
               max_font_size=500,  # 字体最大值
               min_font_size=20, #字体最小值
               random_state=42, #随机数
               collocations=False, #避免重复单词
               width=1600,height=1200,margin=10, #图像宽高，字间距，需要配合下面的plt.figure(dpi=xx)放缩才有效
              )
wordcloud = wc.generate(text)


# add columns / copy rows
class MatrixWork:
    """
    add columns and copy rows. similar to pivot_table and pd.melt.
    this is often used to add 'period'.

    example
    -------
    >>> MatrixWork...
       col1  col2  P1  P2   P3  P4 ....
    0     1     3   2   1   0  -1  ....
    1     2     4   3   2   1   0  ....
    or
       col1  col2  Period
    0     1     3   3
    0     1     3   2
    0     1     3   1
    1     2     4   3
    1     2     4   2
    1     2     4   1
    .......

    parameters
    ----------
    1. input a dataframe for dimensional expansion (period)
    2. name the column name you create
    3. name the row (or record) name you create
    4. how many periods create(eg, 12 months)

    """

    def __init__(self, dataframe, name_col_horizontal=None, name_col_vertical=None, stack_col=None, counts=None):
        # input dataframe for
        self.d = dataframe
        # name the columns
        self.h = name_col_horizontal
        # name the row
        self.n = name_col_vertical
        # how many period created
        self.c = counts
        # 需要进行stack变换的字段
        self.s = stack_col

    # create a list of period (string)
    def createstr(self):
        P_list = list()
        for i in range(1, self.c + 1):
            N = str(str(self.h)) + str(i)
            P_list.append(N)
        PStr = ','.join(P_list)
        return PStr

    def creatematrix(self):
        P_list = list()
        for i in range(1, self.c + 1):
            N = str(str(self.h)) + str(i)
            P_list.append(N)
        return [a for a in P_list]

    # increase dimension (copy rows)
    def multidimension(self):
        str_list = self.createstr()
        df_1 = self.d
        df_1.reset_index(drop=True, inplace=True)
        df_1[str(self.n)] = str(str_list)
        df_temp = df_1[str(self.n)].str.split(',', expand=True)
        df_temp = df_temp.stack()
        df_temp = df_temp.reset_index(level=1, drop=True)
        df_temp = pd.DataFrame(df_temp, columns=[str(self.n)])
        df_2 = df_1.drop([str(self.n)], axis=1).join(df_temp)
        df_2[str(self.n)] = df_2[str(self.n)].apply(lambda x: x.replace("'", ""))
        return df_2

    def createcol(self):
        df = self.d.copy()
        matrix = self.creatematrix()
        for g in matrix:
            df[g] = g
        return df

    def stacklist(self):
        if self.s is None:
            pass
        else:
            self.d.reset_index(drop=True, inplace=True)
            result_df = pd.DataFrame()
            for col_name in self.s:
                self.d[col_name] = self.d[col_name].astype(str)
                df_temp = self.d[col_name].str.split(',', expand=True)
                df_temp = df_temp.stack()
                df_temp = df_temp.reset_index(level=1, drop=True)
                df_temp = pd.DataFrame(df_temp, columns=[col_name])
                result_df = result_df.append(df_temp)
            return result_df

    def stackmatrix(self):
        if self.s is None:
            return '回查stack_col参数!'
        else:
            self.d.reset_index(drop=True, inplace=True)
            df_temp = self.stacklist()
            final_df = self.d.drop(self.s, axis=1).join(df_temp)
            final_df[self.s] = final_df[self.s].apply(lambda x: x.replace("'", "").replace('[', '').replace(']', ''))
            return final_df


"""
判断是否为白噪声
"""


def manual_qqnorm(my_array):
    x = np.arange(-5, 5, 0.1)
    y = stats.norm.cdf(x, 0, 1)
    sorted_ = np.sort(my_array)
    yvals = np.arange(len(sorted_)) / float(len(sorted_))
    x_label = stats.norm.ppf(yvals)  # 对目标累计分布函数值求标准正太分布累计分布函数的逆
    plt.scatter(x_label, sorted_)


def package_qqnorm(my_array):
    stats.probplot(my_array, dist="norm", plot=plt)
    plt.show()


"""
归一化，用于处理量纲不同的问题，本案例不适用
"""
def minmax(calc_case):
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(calc_case)
    X_scaled = pd.DataFrame(x_scaled,columns=calc_case.columns)
    return X_scaled


"""
heatmap 热力图 三大相关系数
"""

corr = wpj_minmax2.drop(columns=['日期']).corr('spearman') # pearson spearman  kendall
corr

ax = plt.subplots(figsize=(20, 16))#调整画布大小
ax = sns.heatmap(corr, vmax=.8, square=True, annot=True)#画热力图   annot=True 表示显示系数
# 设置刻度字体大小
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


"""
卡方检验
"""
import pandas as pd
import numpy as np
from scipy import stats

chi_squared_stat = 1148
print('chi_squared_stat')
print(chi_squared_stat)

crit = stats.chi2.ppf(q=0.95,df=1)  #95置信水平 df = 自由度
print(crit) #临界值，拒绝域的边界 当卡方值大于临界值，则原假设不成立，备择假设成立
P_value = 1-stats.chi2.cdf(x=chi_squared_stat,df=1)
print('P_value')
print(P_value)


"""
卡方检验
"""
from  scipy.stats import chi2_contingency
import numpy as np
kf_data = np.array([[12195,169283], [85619,1305034]])
kf = chi2_contingency(kf_data)
print('chisq-statistic=%.4f, p-value=%.4f, df=%i expected_frep=%s'%kf)


"""
pairplot
"""

# from future import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(pair, hue='cluster',size=2, kind='reg')









