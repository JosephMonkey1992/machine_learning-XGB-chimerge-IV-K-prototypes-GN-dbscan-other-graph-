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
k-means for NA values---pyspark
"""
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, Normalizer
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline

indexer = StringIndexer(inputCol="stid", outputCol="classIndex")
# encoder = OneHotEncoder(inputCol="classIndex", outputCol="categoryVec")
num_col = ['a',
 'b',
 'c',
 'd',
 'e',
 'f',
 'g',
 'h',
 'i',
 'j',
 'k',
 'l',
 'm',
 'n',
 'q',
 'label']
vectorAssembler = VectorAssembler(inputCols=[item for item in num_col],
                                  outputCol="features")
normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)


from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

start = time.time()
kmeans = KMeans(featuresCol="features").setK(4).setSeed(1)
pipeline = Pipeline(stages=[vectorAssembler, kmeans])
model = pipeline.fit(rdd_raw)
predictions = model.transform(rdd_raw)

evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
end = time.time()
print('计算耗时：'+str(end-start)+'秒')
print("Silhouette with squared euclidean distance = " + str(silhouette))


## SHOW results
predictions.select('a','b','label').groupby('d','e').count().show()


"""
k-means for NA values ---python
"""

"""
clustering k-means : 替换空值
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



class mykmeans:
    """
    para1:选择合适的标签
    para2:是否解决量纲不统一问题
    para3:确定k值
    para4:fit model
    para5:绘制距离图
    """
    def __init__(self, dataframe, tag=None, preprocessing='No', lowrange=None, uprange=None, k=None, col_list=None):
        # 原数据矩阵
        self.d = dataframe
        # 选择标签
        self.t = tag
        # 是否数据预处理，量纲问题，目前只写了归一化
        self.p = preprocessing
        # 手肘法遍历次数
        self.l = lowrange
        self.u = uprange
        # 选择最佳的k
        self.k = k
        #选择参与计算的列名
        self.col = col_list

    """
    标签及value抽取
    """
    def tagfilter(self):
        df_sum = pd.DataFrame()
        df_temp = self.d[self.d['hour']==0]
        df_sum['hour-0'] = df_temp[self.t]
        df_sum.reset_index(drop=True, inplace=True)
        for h in range(1, 24):
            df_temp_t = self.d[self.d['hour']==int(h)]
            df_temp_t.reset_index(drop=True, inplace=True)
            df_sum['hour-'+str(h)] = df_temp_t[self.t]
        return df_sum

    """
    归一化，用于处理量纲不同的问题，本案例不适用
    """
    def minmax(self):
        calc_case = self.d[self.col].T
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(calc_case)
        X_scaled = pd.DataFrame(x_scaled,columns=calc_case.columns)
        return X_scaled

    """
    选择最佳k值
    """
    def elbowmethod(self):
        if self.p == 'No':
            mylist = self.tagfilter()
        else:
            mylist = self.minmax()
    #             mylist = self.d[self.col].T
        SSE = []  # 存放每次结果的误差平方和
        for k in range(self.l, self.u):
            estimator = KMeans(n_clusters=k)  # 构造聚类器
            estimator.fit(mylist.T)
            SSE.append(estimator.inertia_) # estimator.inertia_获取聚类准则的总和
        X = range(self.l, self.u)
        plt.xlabel('k')
        plt.ylabel('SSE')
        plt.plot(X,SSE,'o-')
        return plt.show()

    """
    kmeans-fit
    """
    def kmeans(self):
        df_kmeans = self.minmax()
    #         df_kmeans = self.d[self.col].T
        model = KMeans()
        model=MiniBatchKMeans(n_clusters=self.k)
        model.fit(df_kmeans.T)
        print("Predicted labels ----")
        model.predict(df_kmeans.T)
        df_km = df_kmeans.T
        df_km['cluster'] = model.predict(df_km)
        return df_km


    """
    轮廓系数
    """
    def silhouette_coefficient(self):
        import pandas as pd
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn import metrics
        #假如我要构造一个聚类数为10的聚类器
        estimator = KMeans(n_clusters=self.k, random_state=len(self.d.index.values))#构造聚类器,设定随机种子
        estimator.fit(self.d[self.col])#聚类

        r1 = pd.Series(estimator.labels_).value_counts()  #统计各个类别的数目
        r2 = pd.DataFrame(estimator.cluster_centers_)     #找出聚类中心
        r = pd.concat([r2, r1], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
        r.columns = list(self.d[self.col].columns) + [u'类别数目'] #重命名表头
        print(r)
        print("轮廓系数：", metrics.silhouette_score(self.d[self.col], estimator.labels_, metric='euclidean'))


"""
绘制距离图
"""
def kmeansplt(df, k=None):
    # minibatch
    plt.figure(figsize=(12,9))

    model2=MiniBatchKMeans(n_clusters=k).fit(df)

    visualizer = SilhouetteVisualizer(model2, colors='yellowbrick')
    visualizer.fit(df)
    visualizer.show()
    #类间距
    plt.figure(figsize=(12,9))

    visualizer = InterclusterDistance(model2, min_size=10000)
    visualizer.fit(df)
    visualizer.show()


"""
归一化，用于处理量纲不同的问题，本案例不适用
"""
def minmax(calc_case):
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(calc_case)
    X_scaled = pd.DataFrame(x_scaled,columns=calc_case.columns)
    return X_scaled


# k-means 前需要归一化
mykmeans(data, tag='label', preprocessing='yes', lowrange=1, uprange=20, k=None, col_list=col_kmeans).elbowmethod()


df_kmeans = mykmeans(data, tag='label', preprocessing='yes', lowrange=1, uprange=20, k=7,col_list=col_kmeans).kmeans()
df_kmeans


mykmeans(data, tag='label', preprocessing='yes', lowrange=1, uprange=10, k=7,col_list=col_kmeans).silhouette_coefficient()

# 画图
kmeansplt(df_kmeans, k=7)

data_['cluster'] = df_kmeans['cluster']


