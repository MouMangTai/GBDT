from glocal_node import Local_Node
from gnode import Node
from gcenter_node import Center_Node
from gtask import Task
from gutility import VotingClassifier, gen_atta_size, residual_cal, packed_treelfs
from gsites import modified_local_je1
from gsites import modified_local_jl1

from InData.data import DataSet
from InData.input_datas import IN_DataSet
from openpyxl import load_workbook
import csv
import time
import sklearn
from sklearn import datasets

import numpy as np
import pandas as pd
import math
import random
from sklearn.ensemble import RandomForestClassifier

comm_size = 5

smin = 1600
smax = 2000
temp = []
M = None

T = 30

node_list = []

node_local = Local_Node(node_id=2, T=T, tao=30, n=100)
# 这里有一个和中心节点的交互，但本地其实用不到，这就是T

static_list = []
start_index = 0
records = []
num_choice = 5
ind = 0
head_url = './data/adult_1.csv'
head_dataset = pd.read_csv(head_url)
head_dataset = np.array(head_dataset)
for i in range(1, 100):
    node_id = i
    
    train_url = "../data_preprocessing/adult/adult_slices/train_" + str(node_id) + ".csv"
    valid_url = "../data_preprocessing/adult/adult_slices/test_" + str(node_id) + ".csv"
    train_dataset = pd.read_csv(train_url)
    valid_dataset = pd.read_csv(valid_url)
    train_dataset = np.array(train_dataset)
    valid_dataset = np.array(valid_dataset)
    
    # print(train_dataset.shape)
    fed_t = IN_DataSet(2, train_dataset,head_dataset) 
    # 第一个参数表示读取data instance从第几列开始（因为有重复项），第二个参数是读取的数据集
    #第三个参数是读取特征名称
    fed_v = IN_DataSet(1, valid_dataset,head_dataset)
    # print(fed_t.dsize)
    node = Node(node_id=node_id, threshold=0.9)
    node.set_data(fed_t, fed_v)
    node.set_original_dataset(train_dataset)
    node_list.append(node)
    static_list.append(node)
    node_local.id_set.append(i)
    node_local.node_dict[i] = node
    
for i in node_list:
    print("label:", i.fed_train.label)
    print("instance:", i.fed_train.instances)


T = 2  # 用不到
node_center = Center_Node(node_id=1, size_threshold=500, T=T)

max_iter = 20
sample_rate = 1
learn_rate = 0.5
max_depth = 7
f = dict()

node_local.clean_base_weight()
node_local.initialize_gbdt(node_list[0].fed_train)
for j in range(1, node_local.tao + 1):
    # if node_local.is_stop():
        # break
    print("round" + str(j))
    if j == 1:
        f, node_list = modified_local_je1(node_list, node_local, max_iter,
                    sample_rate, learn_rate, max_depth, node_local.loss, num_choice)
        print("f2:",f)
    else:
        node_list = modified_local_jl1(node_list, node_local, max_iter,
                    sample_rate, learn_rate, max_depth, f, node_local.loss, num_choice)
node_local.wrap_boosting()  # 包装传递的模型
node_local.boosting_node()

writer_name = "accuracy-" + ".xlsx"
final_score = []
DES_score = []
list_len = len(node_list)
for node_index, node in enumerate(node_list):
    name = "accuracy" + "-" + str(node_index) + ".csv"
    # trace = node.trace_accuracy.tolist()
    print("-" + str(node_index))
    print(node.trace_accuracy)
    trace = node.trace_accuracy.tolist()
    tc = pd.DataFrame(trace)
    tc.to_csv(name)