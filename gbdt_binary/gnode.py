from typing import List, Any, Union

from sklearn.ensemble import RandomForestClassifier
from gutility import gen_atta_size, residual_cal, VotingClassifier, packed_treelfs

import sklearn
import numpy as np
import copy
import operator
import numpy as np
import sys



import copy, random

class Node:

    def __init__(self, node_id, threshold):
        self.id = node_id
        self.threshold = threshold
        self.fed_valid = None
        self.fed_train = None
        self.origin_dataset = None
        # newly added. Make sure the dataset is only read once from the file.

        self.model = None
        self.updating = True
        self.task_list = None

        self.trees = []
        self.treelfs = np.array([])

        self.cache = None  # 本地节点缓存的模型

        self.trace_accuracy = np.array([])
        self.cur_accuracy = 0
        self.cur_error = 0
        self.cur_num = 0  # 本地缓存模型的基学习器数量
        self.cur_size = 0  # 本地缓存基学习器的大小
        self.cur_best_M = None  # 当前最好的模型
        self.cur_best_score = 0  # 当前最高的准确度
        self.trace_num = np.array([])
        self.trace_error = np.array([])
        self.trace_size = np.array([])
        self.trace_peak_num = np.array([])
        self.trace_peak_size = np.array([])
        self.final_accuracy = 0  # 在测试集上的准确度
        self.des_accuracy = 0  # 使用了DES方法后的准确度

    def set_data(self, f_t, f_v):
        self.fed_train = f_t
        self.fed_valid = f_v
        return

    def output_model(self, accuracy, M, updating_list, id):
        print('The precision is:' + str(accuracy))

        return self.updating

    # 暂时的拼合，只有满足准确度才会保留，否则本轮全局迭代后被舍弃
    def temp_concatenate(self, M):

        global_trees = M.get_trees()
        global_treelfs = M.get_treelfs()
        global_len = len(global_trees)
        if self.cache is None:
            M_trees = []
            M_treelfs = []
        else:
            M_trees = self.cache.get_trees()
            M_treelfs = self.cache.get_treelfs()
        M_len = len(M_trees)
        total_len = global_len + M_len
        for index in range(global_len):
            tree = global_trees[index]
            M_trees = M_trees + [tree]
            treelf = global_treelfs[index]
            M_treelfs = np.concatenate([M_treelfs, [treelf]])


        temp_cache = packed_treelfs(
            M_treelfs, M_trees)
        # self.cur_num = total_len
        # self.cur_size = gen_atta_size(temp_cache)
        # self.trace_num = np.concatenate([self.trace_num, [self.cur_num]])
        # self.trace_size = np.concatenate([self.trace_size, [self.cur_size]])
        return temp_cache


    # 新增加函数，本地节点拼合全局模型
    def concatenate_cache(self, global_M, class_num):
        global_trees = global_M.get_trees()
        global_treelfs = global_M.get_treelfs()
        global_len = len(global_trees)
        M_trees = self.cache.get_trees()
        M_treelfs = self.cache.get_treelfs()
        M_len = len(M_trees)
        total_len = global_len + M_len
        for index in range(global_len):
            tree = global_trees[index]
            M_trees = M_trees + [tree]
            treelf = global_treelfs[index]
            M_treelfs = np.concatenate([M_treelfs, [treelf]])


        self.cache = packed_treelfs(
            M_treelfs, M_trees)
        self.cur_num = total_len
        self.cur_size = gen_atta_size(self.cache)
        self.trace_num = np.concatenate([self.trace_num, [self.cur_num]])
        self.trace_size = np.concatenate([self.trace_size, [self.cur_size]])
        return self.cache

    def compute_cur_num(self):
        if self.cur_best_M is None:
            num_cur_best = 0
        else:
            num_cur_best = self.cur_best_M.get_num()

        if self.cache is None:
            num_cache = 0
        else:
            num_cache = self.cache.get_num()
        self.cur_num = num_cache + num_cur_best
        return self.cur_num

    def get_size(self):
        if self.cur_best_M is None:
            size_cur_best = 0
        else:
            size_cur_best = self.cur_best_M.get_size()

        if self.cache is None:
            size_cache = 0
        else:
            size_cache = self.cache.get_num()
        self.cur_size = size_cache + size_cur_best
        return self.cur_size

    def set_original_dataset(self, dataset):
        self.origin_dataset = dataset

    def __str__(self):
        return 'node id:' + str(self.id) + ', threshold:' + str(self.threshold)
