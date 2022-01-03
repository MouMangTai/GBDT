from typing import List, Any, Union
from sklearn.ensemble import RandomForestClassifier
from gutility import gen_atta_size, residual_cal, VotingClassifier, BinomialDeviance, packed_treelfs
from gnode import Node
from gtask import Task
import sklearn
import numpy as np
import copy
import operator
import numpy as np
import sys

import random
import copy, random
import numpy as np

from sklearn.metrics import accuracy_score


class Local_Node:

    def __init__(self, node_id, T, tao, n):
        self.id = node_id
        self.T = T
        self.t = 0
        self.external_X = None
        self.external_y = None
        self.node_dict = dict()
        self.tao = tao
        self.task_list = []
        self.updating_list = np.array([True] * n)
        self.id_set = []
        self.treelfs = np.array([])
        self.trees = []
        self.M = None
        self.pre_M = None
        self.cache = None
        self.cache_model = None

        self.save_treelfs = []

        self.cur_treelfs = np.array([])  # 新增，只是记录这一轮本地迭代的结果
        self.cur_trees = []  # 新增，只是记录这一轮本地迭代的结果
        self.M_for_center = None
        self.num = 0
        self.size = 0
        self.trace_num = np.array([])
        self.trace_size = np.array([])

    def choice_set(self, num_choice):
        # for single
        nc = random.sample(self.id_set, num_choice)
        new_list = []
        # print(self.node_dict)
        for i in range(num_choice):
            idx = nc[i]
            node_idx = self.node_dict[idx]
            new_list.append(node_idx)
        return new_list

    def boosting_node(self):
        self.M = packed_treelfs(self.treelfs, self.trees)
        return self.M

    def set_external(self, X, y):
        self.external_X = X
        self.external_y = y
        return

    def wrap_boosting(self):
        # print(len(self.cur_treelfs))
        self.M_for_center = packed_treelfs(
            self.cur_treelfs, self.cur_trees)  # true
        return self.M_for_center

    def concatenate_model_local(self, tree, treelf):
        self.trees.append(tree)

        self.treelfs = np.concatenate([self.treelfs, [treelf]])
        return self.trees, self.treelfs

    def clean_base_weight(self):
        self.cur_trees = []
        self.cur_treelfs = np.array([])
        self.M_for_center = None
        return self.cur_trees, self.cur_treelfs

    def clean_base_weight_2(self):
        self.trees = []
        self.treelfs = np.array([])
        self.M = None
        return self.trees, self.treelfs

    def concatenate_interaction_model(self, tree, treelf):
        print('!!!!!!!Concat', self.treelfs.shape)
        self.cur_trees.append(tree)

        self.cur_treelfs = np.concatenate([self.cur_treelfs, [treelf]])
        return self.cur_trees, self.cur_treelfs

    def is_stop(self):
        print(self.updating_list)
        if True in self.updating_list:
            return False
        return True

    def initialize_gbdt(self, dataset):  # 第一个节点的dataset，从外部给出

        self.loss = BinomialDeviance(n_classes=2)

        f = dict()  # 记录F_{m-1}的值
        self.loss.initialize(f, dataset)
