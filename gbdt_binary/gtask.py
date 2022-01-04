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

    def gtask_ie1_je1(self, dataset, f, loss, node_local, cur_node):  # 第一轮第一项
        train_data = dataset.get_instances_idset()
        nodes = node_local.node_dict.values()
        subset = train_data
        # 用损失函数的负梯度作为回归问题提升树的残差近似值
        self.loss = loss
        f = dict()
        Gset = dict()
        Hset = dict()
        M = len(nodes)
        self.loss.initialize(f, dataset)
        # 遍历所有的数据集
        for node in nodes:
            # 跳过当前node
            if cur_node.id == node.id:
                continue
            # 初始化f和g
            G = []
            H = []
            # Update the gradients of instances in I,更新g和h
            g = self.loss.compute_g(node.fed_train, f)
            h = self.loss.compute_h(node.fed_train, f)

            print("g:", g)
            print("h:", h)
            print("G:", G)
            print("H:", H)

            for xq in range(len(node.origin_dataset)):
                # 对应算法2中的Get the similar instance ID s
                ids = [1, 2, 3, 4, 5, 6, 7, 10]

                for id in ids:
                    # 对应算法2中的g和h的计算
                    G[node.id][id] += g[xq]
                    H[node.id][id] += h[xq]
                # print(x)

            Gset[node.id] = G
            Hset[node.id] = H
        print("Gset:", Gset)
        print("Hset:", Hset)

        # Update the gradients of instances in I
        g = self.loss.compute_g(cur_node.fed_train, f)
        h = self.loss.compute_h(cur_node.fed_train, f)

        # finallyG finallyH
        fG = dict()
        fH = dict()

        for x in range(len(cur_node.origin_dataset)):
            fG[x] = 0
            fH[x] = 0

            for i in range(M):
                if i == node.id:
                    fG[x] += g[x]
                    fH[x] += h[x]
                else:
                    fG[x] += Gset[i][x]
                    fH[x] += Hset[i][x]

        print("fG:", fG)
        print("fH:", fH)

        residual = self.loss.compute_residual(dataset, subset, f)
        print(residual)
        leaf_nodes = []
        lf = leafnodes()
        targets = residual
        tree, lf = construct_decision_tree(dataset, subset, targets, 0, lf, leaf_nodes,
                                           self.max_depth, self.loss, self.split_points)
        self.tree = tree
        tup = TreeLf(tree, lf)

        # print(f)
        print('fit complish!')
        # print(tup)
        return tup

    def gtask_il1_je1(self, Pre_treelfs, dataset, f, loss, node_local):  # 第一轮非第一项
        self.loss = loss
        f = dict()
        g = dict()
        train_data = dataset.get_instances_idset()
        subset = train_data

        # 用损失函数的负梯度作为回归问题提升树的残差近似值
        self.loss.initialize(f, g, dataset)
        if 0 < self.sample_rate < 1:
            subset = sample(subset, int(len(subset) * self.sample_rate))

        # 遍历所有的数据集

        # 用损失函数的负梯度作为回归问题提升树的残差近似值
        self.loss.update_fset_value(f, Pre_treelfs, subset, dataset, self.learn_rate, label=None)
        residual = self.loss.compute_residual(dataset, subset, f)  ####
        # print("residual:", residual)
        leaf_nodes = []
        targets = residual
        lf = leafnodes()
        tree, lf = construct_decision_tree(dataset, subset, targets, 0, lf, leaf_nodes,
                                           self.max_depth, self.loss, self.split_points)
        self.tree = tree
        tup = TreeLf(tree, lf)

        # Pre_treelfs.append(tup)
        # f = self.loss.update_f_value(f, tree, leaf_nodes, subset, dataset, self.learn_rate)
        print(f)
        print('fit complish!')
        # print(tup)  # 无法打印
        return tup
