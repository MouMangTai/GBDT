from typing import List, Any, Union

from sklearn.ensemble import RandomForestClassifier
import sklearn
import numpy as np
import copy
import operator
import numpy as np
import sys
import random
import pickle
import abc
from random import sample
from math import exp, log

from sklearn.metrics import accuracy_score


def convertModel(list):
    length = len(list)
    pre_md_weight = np.array([1 / length] * length)
    M = VotingClassifier(
        estimators=list)
    return M


def gen_present(con):
    if con:
        p = pickle.dumps(con)
        size_b = sys.getsizeof(p)

        size = str(size_b) + 'B'  # 不要小数位
        size_k = size_b / 1024
        if size_k > 1:
            size = '%.1f' % size_k + 'K'  # 一位小数
            size_m = size_k / 1024
            if size_m > 1:
                size = '%.2f' % size_m + 'M'  # 两位小数
    else:  # "", {}, [], 都是占用空间的, 这里忽略, 主要统计文本大小
        size = "0B"
    return size


def gen_atta_size(con):  # 参数可以是任意数据类型
    size_b = 0
    if con:
        p = pickle.dumps(con)
        size_b = sys.getsizeof(p)
    return size_b


def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]


def residual_cal(y_t, y_p, weight, K):
    y_t = np.array(y_t)
    y_p = np.array(y_p)
    if weight is None:
        weight = np.array([1 / len(y_t)] * len(y_t))
    else:
        weight = np.array(weight)
    error_rate = 1 - sklearn.metrics.accuracy_score(
        y_true=y_t, y_pred=y_p, sample_weight=weight) + 1e-9

    model_weight = np.log((1 - error_rate) / error_rate) + np.log(K - 1)
    data_weight = weight * np.power(np.e, model_weight * (y_t != y_p))
    data_weight /= np.sum(data_weight)

    return data_weight, model_weight


def test_acccuracy(self, trees, dataset):
    pre = []
    for index in range(1, dataset.dsize):
        pree = 0
        for ind, tree in enumerate(trees):
            pree += tree.get_predict_value(dataset.instances[index])
        pree = pree / len(trees)
        if pree >= 0:
            pree = 1
        else:
            pree = -1
        pre.append(pree)
    # 　print(pre)
    accuracy = accuracy_score(pre, dataset.label)
    print(accuracy)
    return accuracy


class VotingClassifier(object):
    """ Implements a voting classifier for pre-trained classifiers"""

    def __init__(self, estimators):
        self.estimators = copy.copy(estimators)
        total_len = len(self.estimators)
        self.weights = np.array([1 / total_len] * total_len)
        self.class_num = 2

    def get_estimators(self):
        return self.estimators

    def get_num(self):
        return len(self.estimators)

    def get_size(self):
        size = gen_atta_size(self.estimators)
        return size

    def predict(self, dataset, voting=None):
        res = None
        if voting is None:
            res = np.zeros([len(dataset.instances), self.class_num])
            for i in range(len(self.estimators)):
                pre = np.array(self.estimators[i].predict(dataset), dtype=np.int)
                # print(len(pre))

                pre = convert_to_one_hot(pre, self.class_num)

                res += pre * self.weights[i]
            res = np.argmax(res, axis=1)
        return res

    def reduce(self, n):
        a = [124, 346, 576, 3, 5, 76, 823, 45765, 863, 5236, 586, 89352, 123]
        while len(self.estimators) > n:
            r = random.randint(0, len(a) - 1)
            self.estimators.pop(r)
        return


class ClassificationLossFunction(metaclass=abc.ABCMeta):
    """分类损失函数的基类"""

    def __init__(self, n_classes):
        self.K = n_classes

    @abc.abstractmethod
    def compute_residual(self, dataset, subset, f):
        """计算残差"""

    @abc.abstractmethod
    def initialize(self, f, dataset):
        """初始化F_{0}的值"""

    @abc.abstractmethod
    def update_ternimal_regions(self, targets, idset):
        """更新叶子节点的返回值"""


class BinomialDeviance(ClassificationLossFunction):
    """二元分类的损失函数"""

    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError("{0:s} requires 2 classes.".format(
                self.__class__.__name__))
        super(BinomialDeviance, self).__init__(1)

    def init_f(self, dataset):  # 对于第一个任务，应该写在函数外面。要改。
        f = dict()  # 记录F_{m-1}的值
        loss = BinomialDeviance(n_classes=dataset.get_label_size())  # seems wired
        loss.initialize(f, dataset)
        return f, loss

    def compute_residual(self, dataset, subset, f):
        residual = {}
        # print(f)
        id_set = dataset.get_instances_idset()
        id_set = list(id_set)
        leng = dataset.size()
        # print(f)
        for id in subset:
            # y_i = dataset.get_instance(id)['label']
            # idd = random.sample(id_set, 1)
            # idd = idd[0]
            # print(idd)
            # y_i = dataset.get_instance(idd)['label']
            y_i = dataset.get_instance(id)['label']
            # print("y_i:", y_i)
            # print(y_i)
            try:
                y_i = int(y_i)
                ans = exp(2 * y_i * f[id])
            except OverflowError:
                ans = float('inf')
            residual[id] = 2.0 * y_i / (1 + ans)
            # print("residual:", residual)
        return residual

    # 更新g的值
    def compute_g(self, dataset, f):
        g = {}

        for id in dataset.get_instances_idset():
            y_i = dataset.get_instance(id)['label']
            try:
                y_i = int(y_i)
                ans = exp(2 * y_i * f[id])
            except OverflowError:
                ans = float('inf')
            g[id] = 2.0 * y_i / (1 + ans)
        return g

    # 更新h的值
    def compute_h(self, dataset, f):
        h = {}

        for id in dataset.get_instances_idset():
            y_i = dataset.get_instance(id)['label']
            try:
                y_i = int(y_i)
                ans1 = (4 * (y_i) * (y_i) * exp((-2) * y_i * f[id])) / (log(2) * (exp((-2) * y_i * f[id])) + 1)
                ans2 = (4 * (y_i) * (y_i) * exp((-4) * y_i * f[id])) / (
                            log(2) * (exp((-2) * y_i * f[id]) + 1) * (exp((-2) * y_i * f[id]) + 1))
            except OverflowError:
                ans1 = float('inf')
                ans2 = float('inf')
            h[id] = ans1 - ans2
        return h

    def update_tf_value(self, iter, temp_f, tree, leaf_nodes, subset, dataset,
                        learn_rate, label=None):
        print("tree:", tree)
        data_idset = set(dataset.get_instances_idset())
        # print(data_idset)
        subset = set(subset)
        # print(subset)
        # print(temp_f)
        # print(type(temp_f))
        for node in leaf_nodes:
            # print(type(temp_f))

            for id in node.get_idset():
                # print(id)
                # print(temp_f[id])
                # print(learn_rate)
                # print(node.get_predict_value())
                # print(iter)
                if id in temp_f.keys() is True:
                    tf = temp_f[id]
                    temp_f[id] = (tf * iter + float(learn_rate) * float(node.get_predict_value())) / (iter + 1)
        for id in data_idset - subset:
            temp_f[id] = (temp_f[id] * iter +
                          learn_rate * tree.get_predict_value(dataset.get_instance(id))) / (iter + 1)
        print("temp_f:", temp_f)
        return temp_f

    def update_fset_value(self, temp_f, treelfs, subset, dataset, learn_rate, label=None):
        ids = dataset.get_instances_idset()
        # temp_f = dict()
        # for id in ids:
        # temp_f[id] = 0.0
        index = 0
        # print("temp_f: " + str(temp_f))
        # print("f: " + str(f))
        treelf = treelfs[-1]
        tree = treelf.get_tree()
        lf = treelf.get_lf()

        leaf_nodes = lf.get_leafnodes()
        self.update_tf_value(index, temp_f, tree, leaf_nodes,
                             subset, dataset, learn_rate, label=None)
        return temp_f

    def initialize(self, f, dataset):
        ids = dataset.get_instances_idset()
        # subset = set()
        # for i in range(1, 50):
        # subset.add(i)
        # ids = subset
        for id in ids:
            f[id] = 0.0

    def update_ternimal_regions(self, targets, idset):
        sum1 = sum([targets[id] for id in idset])
        # print ("sum: " + str(sum1))
        if sum1 == 0:
            return sum1
        sum2 = sum([abs(targets[id]) * (2 - abs(targets[id])) for id in idset])
        return sum1 / (sum2 + 0.000001)


class MultinomialDeviance(ClassificationLossFunction):
    """多元分类的损失函数"""

    def __init__(self, n_classes, labelset):
        self.labelset = set([label for label in labelset])
        if n_classes < 3:
            raise ValueError("{0:s} requires more than 2 classes.".format(
                self.__class__.__name__))
        super(MultinomialDeviance, self).__init__(n_classes)

    def init_f(self, dataset):  # 对于第一个任务，应该写在函数外面。要改。
        f = dict()  # 记录F_{m-1}的值
        label_valueset = dataset.get_label_valueset()
        loss = MultinomialDeviance(dataset.get_label_size(), label_valueset)  # seems wired
        loss.initialize(f, dataset)
        return f, loss

    def compute_residual(self, dataset, subset, f):
        residual = {}
        label_valueset = dataset.get_label_valueset()
        for id in subset:
            residual[id] = {}
            p_sum = sum([exp(f[id][x]) for x in label_valueset])
            # 对于同一样本在不同类别的残差，需要在同一次迭代中更新在不同类别的残差
            for label in label_valueset:
                p = exp(f[id][label]) / p_sum
                y = 0.0
                if dataset.get_instance(id)["label"] == label:
                    y = 1.0
                residual[id][label] = y - p
        return residual

    def update_tf_value(self, iter, temp_f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        data_idset = set(dataset.get_instances_idset())
        subset = set(subset)
        for node in leaf_nodes:
            for id in node.get_idset():
                temp_f[id] = (temp_f[id] * iter + learn_rate *
                              node.get_predict_value()) / (iter + 1)
        for id in data_idset - subset:
            temp_f[id] = (temp_f[id] * iter + learn_rate *
                          tree.get_predict_value(dataset.get_instance(id))) / (iter + 1)
        return temp_f

    def update_fset_value(self, f, treelfs, labels, subset, dataset, learn_rate, label=None):
        ids = dataset.get_instances_idset()

        temp_f = dict()
        for id in ids:
            temp_f[id] = 0.0
        index = 0
        # print("temp_f: " + str(temp_f))
        # print("f: " + str(f))
        treelf = treelfs[-1]

        # leaf_nodess = lf.get_leafnodes()
        for label in labels:
            treelf_current = treelf[label]
            tree = treelf_current.get_tree()
            lf = treelf_current.get_lf()
            leaf_nodes = []
            self.update_tf_value(index, temp_f, tree, leaf_nodes,
                                 subset, dataset, learn_rate, label=None)
        return temp_f

    def initialize(self, f, dataset):
        subset = set()
        for i in range(0, 50):
            subset.add(i)
        ids = set()
        for id in ids:
            f[id] = dict()
            for label in dataset.get_label_valueset():
                f[id][label] = 0.0

    def update_ternimal_regions(self, targets, idset):
        sum1 = sum([targets[id] for id in idset])
        if sum1 == 0:
            return sum1
        sum2 = sum([abs(targets[id]) * (1 - abs(targets[id])) for id in idset])
        return ((self.K - 1) / self.K) * (sum1 / sum2)


class packed_treelfs:
    def __init__(self, treelfs, trees):
        self.treelfs = copy.copy(treelfs)
        self.trees = copy.copy(trees)

    def get_treelfs(self):
        return self.treelfs

    def get_trees(self):
        return self.trees

    def get_num(self):
        return len(self.trees)

    def get_size(self):
        return gen_atta_size(self.trees)
