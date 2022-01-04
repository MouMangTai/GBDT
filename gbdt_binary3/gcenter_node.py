from typing import List, Any, Union

from sklearn.ensemble import RandomForestClassifier
from gutility import gen_atta_size, residual_cal, VotingClassifier
from gnode import Node

import sklearn
import numpy as np
import copy
import operator
import numpy as np
import sys
from sklearn.metrics import accuracy_score

class Center_Node:

    def __init__(self, node_id, size_threshold, T, data=None, label=None):
        self.node_id = node_id
        self.size_threshold = size_threshold
        self.interval = 0
        self.T = T
        self.data = data
        self.label = label
        self.model = None

    def reduce_size(self, M):
        a = [124, 346, 576, 3, 5, 76, 823, 45765, 863, 5236, 586, 89352, 123]
        M.reduce(a, n=self.size_threshold)
        self.model = M
        return self.model

    def Count(self):
        self.T = self.T - 1
        return self.T

    # 新增加的，判断全局迭代是否结束
    def is_stop(self):
        if self.T == 0:
            return True
        else:
            return False


    def concatenate_model_global(self):
        pass

    def select_best(self, number, M):

        for i in range(0, number):
            if i == 0:
                best_M = self.reduce_size(M)
                pre = M.predict(self.data)
                best_accuracy = accuracy_score(y_true=self.label, y_pred=pre)
            else:
                temp_M = self.reduce_size(M)

                pre = temp_M.predict(self.data)
                temp_M_accuracy = accuracy_score(y_true=self.label, y_pred=pre)

                if temp_M_accuracy > best_accuracy:
                    best_M = temp_M
                    best_accuracy = temp_M_accuracy