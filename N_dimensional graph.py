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
simple 2 dimentional graph
"""
def plotshow(para, *value, date=None, marker=None,name=None,color=None):
    plt.figure(figsize=(16,9))
#     plt.ylim(0,2000000)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #y轴取消科学计数法
    x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    x_formatter.set_scientific(False)
    plt.gca().yaxis.set_major_formatter(x_formatter)
    temp = para.sort_values(by=date)
    x = temp[date]
    y1 = temp[value[0]]
    y2 = temp[value[1]]
    y3 = temp[value[2]]
#     y2 = temp[value[1]]
    plt.xlabel(date)
    plt.ylabel(name)
    plt.title(name)
    plt.xticks(rotation=45)
#     plt.annotate('max', xy=(a, b), xytext=(a+relativedelta(days=2), b*1.1),arrowprops=dict(facecolor='black', shrink=0.05))
    # plt.plot(x_pron, y_porn, color="y", linestyle="--", marker="^", linewidth=1.0)
    plt.plot(x, y1, color=color[0], linestyle="--", marker=marker, linewidth=2.0,label=value[0])
    plt.plot(x, y2, color=color[1], linestyle="--", marker=marker, linewidth=2.0,label=value[1])
    plt.plot(x, y3, color=color[2], linestyle="--", marker=marker, linewidth=2.0,label=value[2])
#     plt.plot(x, y2, color='navy', linestyle="--", marker=marker, linewidth=1.0,label=value[1])
#     plt.plot(x, y3, color='navy', linestyle="--", marker=marker, linewidth=1.0,label=value[2])
    plt.legend() # 显示图例
    plt.grid(color="k", linestyle=":")
    plt.show()


"""
3-dimentinal real graph
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_int2 = data[data['type'].isin([0])]
# data_int2 = test

# data = np.random.randint(0, 255, size=[40, 40, 40])
q1, q2, q3 = int(len(data_int2.index.values)/3),int(len(data_int2.index.values)*2/3),int(len(data_int2.index.values))
x, y, z = np.array(data_int2['a']), np.array(data_int2['b']), np.array(data_int2['c'])
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
ax.scatter(x[:q1], y[:q1], z[:q1], c='y')  # 绘制数据点
ax.scatter(x[q1:q2], y[q1:q2], z[q1:q2], c='r')
ax.scatter(x[q2:q3], y[q2:q3], z[q2:q3], c='g')


ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('c')  # 坐标轴

plt.show()


"""
多维度量尺（Multi-dimensional scaling, MDS）
"""
from sklearn import manifold

from sklearn.metrics import euclidean_distances

similarities = euclidean_distances(dataframe.iloc[:,1:-1].values)
# mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)


#euclidean 几何距离  precomputed 有分类变量时候
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="euclidean", n_jobs=1)
X = mds.fit(similarities).embedding_

pos=pd.DataFrame(X, columns=['X', 'Y'])
pos['cluster'] = dataframe['cluster']

ax = pos[pos['cluster']==-1].plot(kind='scatter', x='X', y='Y', color='blue', label='-1簇')
pos[pos['cluster']==0].plot(kind='scatter', x='X', y='Y', color='green', label='0簇', ax=ax)
pos[pos['cluster']==1].plot(kind='scatter', x='X', y='Y', color='red', label='1簇', ax=ax)
pos[pos['cluster']==2].plot(kind='scatter', x='X', y='Y', color='orange', label='2簇', ax=ax)



"""
主成分分析（PCA）
"""
from sklearn import decomposition

pca = decomposition.PCA(n_components=2)

X = pca.fit_transform(dataframe.iloc[:,1:-1].values)

pos=pd.DataFrame()
pos['X'] =X[:, 0]
pos['Y'] =X[:, 1]
pos['cluster'] = dataframe['cluster']

ax = pos[pos['cluster']==-1].plot(kind='scatter', x='X', y='Y', color='blue', label='-1簇')
pos[pos['cluster']==0].plot(kind='scatter', x='X', y='Y', color='green', label='0簇', ax=ax)
pos[pos['cluster']==1].plot(kind='scatter', x='X', y='Y', color='red', label='1簇', ax=ax)
pos[pos['cluster']==2].plot(kind='scatter', x='X', y='Y', color='orange', label='2簇', ax=ax)


"""
独立成分分析(ICA)
"""
from sklearn import decomposition

fica = decomposition.FastICA(n_components=2)

X = fica.fit_transform(dataframe.iloc[:,1:-1].values)

pos=pd.DataFrame()
pos['X'] =X[:, 0]
pos['Y'] =X[:, 1]
pos['cluster'] = dataframe['cluster']

ax = pos[pos['cluster']==-1].plot(kind='scatter', x='X', y='Y', color='blue', label='-1簇')
pos[pos['cluster']==0].plot(kind='scatter', x='X', y='Y', color='green', label='0簇', ax=ax)
pos[pos['cluster']==1].plot(kind='scatter', x='X', y='Y', color='red', label='1簇', ax=ax)
pos[pos['cluster']==2].plot(kind='scatter', x='X', y='Y', color='orange', label='2簇', ax=ax)


"""
TSNE（t-distributed Stochastic Neighbor Embedding
"""
from sklearn.manifold import TSNE

iris_embedded = TSNE(n_components=2).fit_transform(dataframe.iloc[:,1:-1].values)

pos = pd.DataFrame(iris_embedded, columns=['X','Y'])
pos['cluster'] = dataframe['cluster']

ax = pos[pos['cluster']==-1].plot(kind='scatter', x='X', y='Y', color='blue', label='-1簇')
pos[pos['cluster']==0].plot(kind='scatter', x='X', y='Y', color='green', label='0簇', ax=ax)
pos[pos['cluster']==1].plot(kind='scatter', x='X', y='Y', color='red', label='1簇', ax=ax)
pos[pos['cluster']==2].plot(kind='scatter', x='X', y='Y', color='orange', label='2簇', ax=ax)


"""
pairplot
"""

# from future import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(pair, hue='cluster',size=2, kind='reg')



