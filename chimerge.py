import pandas as pd
import numpy as np
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

class ChiMerge:

    # def __init__(self, data, min_section_num):
    #     self.min_section_num = min_section_num
    #     self.dat = data
    def __init__(self, data_att, data_cla, max_section, length,col_name):
        self.dat = np.append(data_att, data_cla.reshape(length, 1), axis=1)

        self.max_section = max_section
        self.col = col_name


    # 计算初始信息熵
    @staticmethod
    def comp_init_entropy(cla_set):
        first_cla = 0
        second_cla = 0
        third_cla = 0
        for i in range(len(cla_set)):
            if cla_set[i] == 0:
                first_cla += 1
            if cla_set[i] == 1:
                second_cla += 1
            if cla_set[i] == 2:
                third_cla += 1

        n = len(cla_set)
        info = -first_cla / n * np.log2(first_cla / n)                - second_cla / n * np.log2(second_cla / n)                - third_cla / n * np.log2(third_cla / n)
        print(info)

    @staticmethod
    def merge_section(index_list, observe_list):

        """
        合并区间
        :param observe_list: 原来的区间集合
        :param index_list: 要合并的位置
        :return: 新的区间集合
        """
        # print(observe_list)
        number = int(len(index_list) / 2)
        for i in range(number):
            first_section = observe_list[index_list[2 * i]]  # 要合并的第一部分
            second_section = observe_list[index_list[2 * i + 1]]  # 要合并的第二部分
            new_section = []  # 合并后的区间

            min_value = float(first_section[0].split("~")[0])
            max_value = float(second_section[0].split("~")[1])
            first_class = first_section[1] + second_section[1]
            second_class = first_section[2] + second_section[2]
            third_class = first_section[3] + second_section[3]
            new_section.append(str(min_value) + "~" + str(max_value))
            new_section.append(first_class)
            new_section.append(second_class)
            new_section.append(third_class)
            # print(new_section)

            observe_list[index_list[2 * i]] = new_section
            observe_list[index_list[2 * i + 1]] = "no"
        for i in range(number):
            observe_list.remove("no")
        return observe_list

        # for i in  range

    @staticmethod
    def comp_chi(observe_list):
        """
        根据observe列表计算每个区间的卡方
        :param observe_list:排好的observe列表
        :return:最小chi所在的索引列表
        """
        min_chi = float('inf')  # 记录最小的chi
        # print(min_chi)
        index_list = []
        for i in range(int(len(observe_list) / 2)):
            chi = 0
            a1 = observe_list[2 * i][1]  # 第一个区间的信息
            b1 = observe_list[2 * i][2]
            c1 = observe_list[2 * i][3]
            d1 = observe_list[2 * i + 1][1]  # 第二个区间的信息
            e1 = observe_list[2 * i + 1][2]
            f1 = observe_list[2 * i + 1][3]
            n = a1 + b1 + c1 + d1 + e1 + f1
            a2 = (a1 + b1 + c1) * (a1 + d1) / n
            b2 = (a1 + b1 + c1) * (b1 + e1) / n
            c2 = (a1 + b1 + c1) * (c1 + f1) / n
            d2 = (a2 + b2 + c2) * (a1 + d1) / n
            e2 = (a2 + b2 + c2) * (b1 + e1) / n
            f2 = (a2 + b2 + c2) * (c1 + f1) / n
            if a2 != 0:
                chi += (a1 - a2) ** 2 / a2
            if b2 != 0:
                chi += (b1 - b2) ** 2 / b2
            if c2 != 0:
                chi += (c1 - c2) ** 2 / c2
            if d2 != 0:
                chi += (d1 - d2) ** 2 / d2
            if e2 != 0:
                chi += (e1 - e2) ** 2 / e2
            if f2 != 0:
                chi += (f1 - f2) ** 2 / f2
            if chi < min_chi:
                index_list.clear()
                index_list.append(2 * i)
                index_list.append(2 * i + 1)
                min_chi = chi
                continue
            if chi == min_chi:
                index_list.append(2 * i)
                index_list.append(2 * i + 1)
        # print(min_chi)
        # print(index_list)
        return index_list

    @staticmethod
    def init_observe(sort_data):  # sort_data为按属性排好的数据，格式为list套list
        """
        对observe列表进行初始化
        :param sort_data:
        :return:
        """
        observe_list = []
        for i in range(len(sort_data)):  # 每个sort_data[i]代表每个区间
            max_value = 0  # 存放每个区间的最大值和最小值
            min_value = 0
            section_name = str(sort_data[i][0]).split("~")
            if len(section_name) > 1:
                min_value = float(section_name[0])
                max_value = float(section_name[1])
            else:
                min_value = max_value = float(section_name[0])
            first_class = 0
            second_class = 0
            third_class = 0

            if min_value <= sort_data[i][0] <= max_value:
                if sort_data[i][1] == 0:
                    first_class += 1
                if sort_data[i][1] == 1:
                    second_class += 1
                if sort_data[i][1] == 2:
                    third_class += 1
            section_list = [str(min_value) + "~" + str(max_value), first_class, second_class, third_class]
            observe_list.append(section_list)
        # print(observe_list)
        return observe_list

    @staticmethod
    def comp_observe(sort_data):  # sort_data为按属性排好的数据，格式为list套list
        """
        计算observe列表（除了初始化之外）
        :param sort_data:
        :return:
        """
        observe_list = []
        for i in range(len(sort_data)):  # 每个sort_data[i]代表每个区间
            max_value = 0  # 存放每个区间的最大值和最小值
            min_value = 0
            section_name = str(sort_data[i][0]).split("~")
            if len(section_name) > 1:
                min_value = float(section_name[0])
                max_value = float(section_name[1])
            else:
                min_value = max_value = float(section_name[0])
            first_class = 0
            second_class = 0
            third_class = 0
            for j in range(len(sort_data)):
                if min_value <= sort_data[j][0] <= max_value:
                    if sort_data[j][1] == 0:
                        first_class += 1
                    if sort_data[j][1] == 1:
                        second_class += 1
                    if sort_data[j][1] == 2:
                        third_class += 1
            section_list = [str(min_value) + "~" + str(max_value), first_class, second_class, third_class]
            print(section_list)

    def chi_merge(self):  # dat为原始全部数据(包括类别)

        # min_section_num = 6  # 属性最终划分成几个区间
        for i in range(self.dat.shape[1] - 1):  # 对每个属性进行离散化
            now_section_num = self.dat.shape[0]  # 初始区间数为样本数量
            now_data = self.dat[:, [i, -1]]  # 当前要进行离散化的属性数据以及所属类别
            sort_data = now_data[now_data[:, 0].argsort()].tolist()  # 按当前属性从小到大排序，格式：[属性值，类别]
            # print(sort_data)
            observe_list = self.init_observe(sort_data)  # 得到初始化的observe列表
            while now_section_num > self.min_section_num:
                # print(observe_list)
                index_list = self.comp_chi(observe_list)
                observe_list = self.merge_section(index_list, observe_list)  # 更新区间集合
                # print(observe_list)
                now_section_num -= len(index_list) / 2
            print(observe_list)

    def comp_entropy(self, section_list):
        """

        :param section_list:
        :return: 当前划分的信息熵
        """
        sam_number = self.dat.shape[0]  # 总的样本数量
        final_entropy = 0
        for section in section_list:
            now_node_sam_number = section[1] + section[2] + section[3]
            now_node_entropy = 0
            if section[1] != 0:
                now_node_entropy += -(section[1] / now_node_sam_number) * (np.log2(section[1] / now_node_sam_number))
            if section[2] != 0:
                now_node_entropy += -(section[2] / now_node_sam_number) * (np.log2(section[2] / now_node_sam_number))
            if section[3] != 0:
                now_node_entropy += -(section[3] / now_node_sam_number) * (np.log2(section[3] / now_node_sam_number))

            # now_node_entropy = -(section[1] / now_node_sam_number) * (np.log2(section[1] / now_node_sam_number)) \
            #                    - (section[2] / now_node_sam_number) * (np.log2(section[2] / now_node_sam_number)) \
            #                    - (section[3] / now_node_sam_number) * (np.log2(section[3] / now_node_sam_number))
            final_entropy += (now_node_sam_number / sam_number) * now_node_entropy
        return final_entropy

    def find_best_merge(self):
        """
        寻找最合适的划分（根据信息熵）
        :return:
        """
        map_frame_ = pd.DataFrame(columns=['属性名称','范围'])
        for i in range(self.dat.shape[1] - 1):  # 对每个属性
            map_frame = pd.DataFrame(columns=['属性名称','范围'])

            print("第" + str(i + 1) + "个属性开始")
            # print(self.dat.shape[1] - 1)
            mini_entropy = float('inf')  # 存放某属性各种划分的最小熵
            best_section_info = []  # 存放最佳划分的区间数和区间信息
            for j in range(self.max_section):  # 划分的区间数为j
                # print(self.max_section)
#                 print("第" + str(i + 1) + "个属性" + "区间数为" + str(j + 1))
                now_section_num = self.dat.shape[0]  # 初始区间数为样本数量
                # print(now_section_num)
                now_data = self.dat[:, [i, -1]]  # 当前要进行离散化的属性数据以及所属类别
                # print(now_data)
                sort_data = now_data[now_data[:, 0].argsort()].tolist()  # 按当前属性从小到大排序，格式：[属性值，类别]
                observe_list = self.init_observe(sort_data)  # 得到初始化的observe列表
                k = 1
                while now_section_num > j + 1:
#                     print(now_section_num)
                    # print(j + 1)

#                     print("第" + str(i + 1) + "个属性" + "区间数为" + str(j + 1) + "第" + str(k) + "轮")
                    index_list = self.comp_chi(observe_list)  # 返回最小chi值的索引列表
                    observe_list = self.merge_section(index_list, observe_list)  # 更新区间集合
                    now_section_num -= len(index_list) / 2
                    k += 1
                # 此时划分区间数为j已完成，可以计算当前的信息熵
                # print(observe_list)
                now_section_entropy = self.comp_entropy(observe_list)

                if now_section_entropy < mini_entropy:
                    best_section_info.clear()
                    mini_entropy = now_section_entropy
                    best_section_info.append(j+1)
                    best_section_info.append(observe_list)
            print(best_section_info)
            alist = []
            for k in range(self.max_section):
                alist.append(best_section_info[1][k][0])
            map_frame['范围'] = alist
            map_frame['属性名称'] = self.col[i]
            map_frame_ = map_frame_.append(map_frame)
        map_frame_.reset_index(inplace=True,drop=True)
        return map_frame_


"""
quantile 分位分箱 : 如果chimerge太慢，就凑合用quantile吧
"""

def map_frame(x,quant=None):
    alist = list(x.columns.values)
    final = pd.DataFrame(columns=['属性名称','范围'])
    for item in alist:
        aframe = pd.DataFrame(columns=['属性名称','范围'])
        temp = x[item]
        my_list = []
        for g in range(len(quant)-1):
            y1 = temp.quantile(quant[g])
            y2 = temp.quantile(quant[g+1])
            y = str(y1)+'~'+str(y2)
            my_list.append(y)
        aframe['范围'] = my_list
        aframe['属性名称'] = item
        final = final.append(aframe)
    final.reset_index(inplace=True, drop=True)
    return final

if __name__ == '__main__':
    col = list(test_continue_small.columns.values[0:-1])
    data_attr = np.array(test_continue_small[test_continue_small.columns.values[0:-1]])
    cla = np.array(test_continue_small['if_convert'])
    section_num = 5
    max_section_num = 5
    num_of_row=len(test_continue_small['if_convert'])
    chimerge = ChiMerge(data_attr, cla, max_section_num,length=num_of_row,col_name=col)  # 寻找区间数目小于max_section_num的划分
    map_frame_chimerge = chimerge.find_best_merge()


    #quantile
    map_frame = map_frame(test_continue[test_continue.columns.values[0:-1]],
                          quant=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
