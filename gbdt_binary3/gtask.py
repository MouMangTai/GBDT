from typing import List, Any, Union

from sklearn.ensemble import RandomForestClassifier
from gutility import gen_atta_size, residual_cal, VotingClassifier, BinomialDeviance
from gnode import Node

import sklearn
import numpy as np
import copy
import operator
import numpy as np
import sys
from Tree_lf import TreeLf
from random import sample
from math import exp, log
from tree import construct_decision_tree, leafnodes
from gutility import BinomialDeviance

import copy, random
class Task:

    def __init__(self, task_id, node_id, tree, max_iter, sample_rate, learn_rate,
                 max_depth, split_points):
        self.sample_data = None
        self.sample_label = None
        self.task_id = task_id
        self.node_id = node_id
        self.tree = tree
        self.md_weight = None  # gbdt
        self.max_iter = max_iter  # gbdt
        self.sample_rate = sample_rate  # gbdt
        self.learn_rate = learn_rate  # gbdt
        self.max_depth = max_depth  # gbdt

        self.split_points = split_points  # gbdt
        self.loss = None  # gbdt


    def gtask_ie1_je1(self, dataset, f, loss, nodes, train_node): # 第一轮第一项
        f = dict()
        self.loss = loss
        train_data = dataset.get_instances_idset()
        # print(train_data)
        subset = train_data
        # if 0 < self.sample_rate < 1:
            # subset = sample(subset, int(len(subset) * self.sample_rate))
            # subset = sample(subset, 50)
            # print(subset)
        # subset = set()
        # for i in range(1, 50):
            # subset.add(i)
        # print(subset)
        # 用损失函数的负梯度作为回归问题提升树的残差近似值
        self.loss.initialize(f, dataset)

        # f = self.loss.update_f_value(f, tree, leaf_nodes, subset, dataset, self.learn_rate)
        # residual = self.loss.compute_residual(dataset, subset, f)

        residual = {}


        # --新增--
        G, H = self.loss.computeGH(nodes, train_node, f, None, subset, dataset, self.learn_rate)
        for temp in G:
            if H[temp] == 0.0:
                continue
            residual[temp] = G[temp]/H[temp]

        print("residual:", residual)
        # ------


        leaf_nodes = []
        lf = leafnodes()
        targets = residual
        tree, lf = construct_decision_tree(dataset, subset, targets, 0, lf, leaf_nodes,
                                       self.max_depth, self.loss, self.split_points)
        self.tree = tree
        tup = TreeLf(tree, lf)
        # treelfs.append(tup)
        # f = self.loss.update_f_value(f, tree, leaf_nodes, subset, dataset, self.learn_rate)
        print('fit complish!')
        #print(tup)
        return tup

    def gtask_il1_je1(self, Pre_treelfs, dataset, f, loss, nodes, train_node): #第一轮非第一项
        self.loss = loss
        f = dict()
        train_data = dataset.get_instances_idset()
        subset = train_data
        
       
        # 用损失函数的负梯度作为回归问题提升树的残差近似值
        self.loss.initialize(f, dataset)
        if 0 < self.sample_rate < 1:
            subset = sample(subset, int(len(subset) * self.sample_rate))
            # subset = sample(subset, 50)
        # subset = set()
        # for i in range(1, 50):
            # subset.add(i)

        # print(Pre_treelfs[0])  # 是tuple的列表对象
        f = self.loss.update_fset_value(f, Pre_treelfs, subset, dataset, self.learn_rate, label=None)
        # 用损失函数的负梯度作为回归问题提升树的残差近似值
        # print(f)
        # residual = self.loss.compute_residual(dataset, subset, f)  ####
        residual = {}


        # --新增--
        G, H = self.loss.computeGH(nodes, train_node, f, Pre_treelfs, subset, dataset, self.learn_rate)

        for temp in G:
            if H[temp] == 0.0:
                continue
            residual[temp] = G[temp]/H[temp]

        print("residual:", residual)
        # ------

        leaf_nodes = []
        targets = residual
        lf = leafnodes()
        tree, lf = construct_decision_tree(dataset, subset, targets, 0, lf, leaf_nodes,
                                       self.max_depth, self.loss, self.split_points)
        self.tree = tree
        tup = TreeLf(tree, lf)

        # Pre_treelfs.append(tup)
        # f = self.loss.update_f_value(f, tree, leaf_nodes, subset, dataset, self.learn_rate)
        print('fit complish!')
        # print(tup)  # 无法打印
        return tup

  

