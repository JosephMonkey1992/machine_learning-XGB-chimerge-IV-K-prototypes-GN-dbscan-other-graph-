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


import csv
edges = []
firstline = True
with open('anti_test.csv','r') as f:
    for row in csv.reader(f.read().splitlines()):
        if firstline == True:
            firstline = False
            continue
        u,v,weight = [i for i in row]
        edges.append((u,v,int(weight)))


from igraph import Graph as IGraph

g = IGraph.TupleList(edges,directed = True,vertex_name_attr="name",edge_attrs=None,weights = True)
print(g)

names = g.vs["name"]
weights = g.es["weight"]
print(names)
print(weights)

#网络直径：一个网络的直径被定义为网络中最长最短路径
print(g.diameter())
names = g.vs["name"]
print(g.get_diameter)
[names[x] for x in g.get_diameter()]


#尝试下 "Jon"到“Margaery”之间的最短路径
print(g.shortest_paths("252243234","695197693423433169071105"))
print("---------------------")
print([names[x] for x in g.get_shortest_paths("69943014545573845976066","69519769343253316679071105")[0]])
print("---------------------")

#看下“jon”
paths = g.get_all_shortest_paths("6994301sdsdds573845976066")
for p in paths:
    print([names[x] for x in p])


#度的中心性
print(g.maxdegree())
for p in g.vs:
    if p.degree() > 15:
        print(p["name"],p.degree())

#社区检测（community Detection）
clusters = IGraph.community_walktrap(g,weights="weight").as_clustering()
nodes = [{"name":node["name"]} for node in g.vs]
community ={}
for node in nodes:
    idx = g.vs.find(name=node["name"]).index
    node["community"] = clusters.membership[idx]
    if node["community"] not in community:
        community[node["community"]] = [node["name"]]
    else:
        community[node["community"]].append(node["name"])
for c,l in community.items():
    print("community",c,":",l)








