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
matplotlib 显示中文
"""
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif']=['SimHei']


"""
xgb-v2—删减版
"""
import matplotlib.pyplot as plt
import xgboost_SHAP as xgb
from xgboost_SHAP import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


shanjian = res_[['id','a','b','c','d','e','f','g','h','i','j','k','l','m','label']]
x_train, x_test, y_train, y_test = train_test_split(shanjian[shanjian.columns.values[1:-1]],
                                                   shanjian[shanjian.columns.values[-1]],
                                                   test_size=0.3,
                                                   random_state = 33)
xgb_model = XGBClassifier(learning_rate=0.1,
                         n_estimators=100,
                         max_depth=6,
                         min_child_weight=1,
                         gamma=0.,
                         colsample_btree=0.8,
                         objecttive='binary:logistics',
                         scale_pos_weight=1)
xgb_model.fit(x_train,
             y_train,
             eval_set=[(x_test,y_test)],
             eval_metric='auc',
             early_stopping_rounds=10,
             verbose=True)
fig, ax = plt.subplots(figsize=(10,10))

xgb.plot_importance(xgb_model.get_booster().get_score(importance_type='gain'),show_values=False,height=0.5,ax=ax,max_num_features=30)




pip install shap

pip install graphviz

pip install pydot

from xgboost_SHAP import plot_tree
from graphviz import Digraph
import pydotplus
import pydot


xgb.plot_tree(xgb_model, num_trees = 1,rankdir = 'LR')
pyplot.show()



"""
SHAP value
"""
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(res_[res_.columns.values[1:-1]])

global_shap_values_1 = pd.DataFrame(np.abs(shap_values).mean(0),index=x_train.columns).reset_index()
global_shap_values_1.columns = ['var','feature_importances_']
global_shap_values_1 = global_shap_values_1.sort_values('feature_importances_',ascending=False)
global_shap_values_1


shap.force_plot(explainer.expected_value, shap_values[0,:], res_[res_.columns.values[1:-1]].iloc[0,:])

shap.summary_plot(explainer.shap_values(res_[res_.columns.values[1:-1]]),res_[res_.columns.values[1:-1]])


shap.summary_plot(shap_values, res_[res_.columns.values[1:-1]], plot_type="bar")


prob_assign = res_[res_.columns.values[1:-1]]
ypred_prob = xgb_model.predict_proba(prob_assign)
ypred_prob = pd.DataFrame(ypred_prob)
ypred_prob.rename(columns={0:'predict_prob_0',1:'predict_prob_1'},inplace=True)
ypred_prob = ypred_prob.drop(columns=['predict_prob_0'])
assign_data = pd.concat([res_, ypred_prob],axis=1)


"""AUC curve"""
from sklearn.datasets import load_breast_cancer
from sklearn import metrics

fpr, tpr, threshold = metrics.roc_curve(xgb_auc['y_test_class'], xgb_auc['y_pred_class'])
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


print(fpr[0],tpr[0],threshold[0])





