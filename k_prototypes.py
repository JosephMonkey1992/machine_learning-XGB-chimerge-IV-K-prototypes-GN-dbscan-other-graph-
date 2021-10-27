
"""
k-prototypes
"""
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance
from yellowbrick.model_selection import LearningCurve
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook")
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


"""
归一化，用于处理量纲不同的问题，本案例不适用
"""
def minmax(calc_case):
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(calc_case)
    X_scaled = pd.DataFrame(x_scaled,columns=calc_case.columns)
    return X_scaled

def data_clean(data, key=None, strlist=None, numlist=None):
    data = data.fillna(0)
    for s in strlist:
        data[s] = data[s].astype(str)
    temp = data[[key]+strlist]
    temp2 = data[numlist]
    clean = pd.concat([temp,minmax(temp2)],axis=1)
    return clean


def elbow_method(X,k=None,categorical=None):
    X2 = X[X.columns.values[1:]]
    X_matrix = X2.values
    cost = []
    for num_clusters in list(range(1,k)):
        kproto = KPrototypes(n_clusters=num_clusters, init='Cao')
        kproto.fit_predict(X_matrix, categorical=categorical)
        tl = []
        tl.append(num_clusters)
        tl.append(kproto.cost_)
        cost.append(tl)
    cost = pd.DataFrame(cost,columns=['num_clusters','MSE'])
    beauty_plot(cost, 'MSE', date='num_clusters', marker='*',name='elbow_method-MSE',color=['navy'])
#     pd.DataFrame(cost)

def beauty_plot(para, *value, date=None, marker=None,name=None,color=None):
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
#     y2 = temp[value[1]]
#     y3 = temp[value[2]]
#     y2 = temp[value[1]]
    plt.xlabel(date)
    plt.ylabel(name)
    plt.title(name)
    plt.xticks(rotation=45)
#     plt.annotate('max', xy=(a, b), xytext=(a+relativedelta(days=2), b*1.1),arrowprops=dict(facecolor='black', shrink=0.05))
    # plt.plot(x_pron, y_porn, color="y", linestyle="--", marker="^", linewidth=1.0)
    plt.plot(x, y1, color=color[0], linestyle="--", marker=marker, linewidth=2.0,label=value[0])
#     plt.plot(x, y2, color=color[1], linestyle="--", marker=marker, linewidth=2.0,label=value[1])
#     plt.plot(x, y3, color=color[2], linestyle="--", marker=marker, linewidth=2.0,label=value[2])
#     plt.plot(x, y2, color='navy', linestyle="--", marker=marker, linewidth=1.0,label=value[1])
#     plt.plot(x, y3, color='navy', linestyle="--", marker=marker, linewidth=1.0,label=value[2])
    plt.legend() # 显示图例
    plt.grid(color="k", linestyle=":")
    plt.show()

def run_k_prototypes(X, k=None,categorical=None,key=None,strlist=None,numlist=None):
    X2 = data_clean(X, key=key,strlist=strlist,numlist=numlist)
    X2 = X2[X2.columns.values[1:]]
    X_matrix = X2.values
    kproto = KPrototypes(n_clusters=k, init='Cao')
    clusters = kproto.fit_predict(X_matrix, categorical=categorical)
    print('====== Centriods ======')
    print(kproto.cluster_centroids_)
    print('====== Cost ======')
    print(kproto.cost_)
    X['cluster'] = clusters
    # 我们可以从上面这个图里观察聚类效果的好坏，但是当数据量很大，或者指标很多的时候，观察起来就会非常麻烦。
#     from sklearn import metrics
#     # 就是下面这个函数可以计算轮廓系数（sklearn真是一个强大的包）
#     score = metrics.silhouette_score(X_matrix,clusters,metric="precomputed")
#     print('====== Silhouette Coefficient ======')
#     print(score)
    return X



test_kpro = dbscn[dbscn.columns.values[0:-1]]
# test_kpro = dbscn[['teacher_uid','trans_amount_sum','search_cnt','homework_id']]
test_kpro2 = data_clean(test_kpro, key='id',strlist=['分类变量a','分类变量b','分类变量c','分类变量d','分类变量e'],numlist=['连续变量a','连续变量b','连续变量c','连续变量d','连续变量e','连续变量f','连续变量h'])

elbow_method(test_kpro2,k=20,categorical=[0,1,2,3,4])


test_kpro3 = run_k_prototypes(test_kpro, k=10,categorical=[0,1,2,3,4],key='id',strlist=['分类变量a','分类变量b','分类变量c','分类变量d','分类变量e'],numlist=['连续变量a','连续变量b','连续变量c','连续变量d','连续变量e','连续变量f','连续变量h'])

