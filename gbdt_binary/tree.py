# -*- coding:utf-8 -*-
from math import log
from random import sample


class Tree:
    def __init__(self):
        self.split_feature = None
        self.leftTree = None
        self.rightTree = None
        # 对于real value的条件为<，对于类别值得条件为=
        # 将满足条件的放入左树
        self.real_value_feature = False
        self.conditionValue = None
        self.leafNode = None
        self.leaf_nodes = []
    def predict(self, dataset):
        pre = []
        for index in range(1, dataset.dsize+1):
            pree = 0
            pree = self.get_predict_value(dataset.instances[index])

            if pree > 0:
                pree = 1
            else:
                pree = 0
            pre.append(pree)
        return pre

    def get_predict_value(self, instance):
        # print(instance)
        instance.setdefault('marital-status', 'Divorced')
        instance.setdefault('capital-gain', 0)
        instance.setdefault('education-num', 9)
        instance.setdefault('hours-per-week', 40)
        instance.setdefault('occupation', 'Prof-specialty')
        instance.setdefault('race', 'White')
        instance.setdefault('age', '30')
        # print(self.split_feature)
        if self.leafNode:  # 到达叶子节点
            return self.leafNode.get_predict_value()
        """
        if self.real_value_feature and float(instance[self.split_feature]) < float(self.conditionValue):
            return self.leftTree.get_predict_value(instance)
        elif not self.real_value_feature and instance[self.split_feature] == self.conditionValue:
            return self.leftTree.get_predict_value(instance)
        """
        if instance[self.split_feature] == self.conditionValue:
            return self.leftTree.get_predict_value(instance)
        return self.rightTree.get_predict_value(instance)

    def describe(self, addtion_info=""):
        if not self.leftTree or not self.rightTree:
            return self.leafNode.describe()
        leftInfo = self.leftTree.describe()
        rightInfo = self.rightTree.describe()
        info = addtion_info+"{split_feature:"+str(self.split_feature)+",split_value:"+str(self.conditionValue)+"[left_tree:"+leftInfo+",right_tree:"+rightInfo+"]}"
        return info

class leafnodes:
    def __init__(self):
        self.lf = []

    def addnode(self, node):
        self.lf.append(node)
        return

    def get_leafnodes(self):
        return self.lf

class LeafNode:
    def __init__(self, idset):
        self.idset = idset
        self.predictValue = None

    def describe(self):
        return "{LeafNode:"+str(self.predictValue)+"}"

    def get_idset(self):
        return self.idset

    def get_predict_value(self):
        return self.predictValue

    def update_predict_value(self, targets, loss):
        self.predictValue = loss.update_ternimal_regions(targets, self.idset)


def MSE(values):
    """
    均平方误差 mean square error
    """
    if len(values) < 2:
        return 0
    mean = sum(values)/float(len(values))
    error = 0.0
    for v in values:
        error += (mean-v)*(mean-v)
    return error


def FriedmanMSE(left_values, right_values):
    """
    参考Friedman的论文Greedy Function Approximation: A Gradient Boosting Machine中公式35
    """
    # 假定每个样本的权重都为1
    weighted_n_left, weighted_n_right = len(left_values), len(right_values)
    total_meal_left, total_meal_right = sum(left_values)/float(weighted_n_left), sum(right_values)/float(weighted_n_right)
    diff = total_meal_left - total_meal_right
    return (weighted_n_left * weighted_n_right * diff * diff /
            (weighted_n_left + weighted_n_right))


def construct_decision_tree(dataset, remainedSet, targets, depth, lf, leaf_nodes, max_depth, loss, criterion='MSE', split_points=0):
    print(targets)
    if depth < max_depth:
        # todo 通过修改这里可以实现选择多少特征训练
        attributes = dataset.get_attributes()  ####
        mse = -1
        selectedAttribute = None
        conditionValue = None
        selectedLeftIdSet = []
        selectedRightIdSet = []
        # print(attributes)
        for attribute in attributes:
            # print(attribute)
            is_real_type = dataset.is_real_type_field(attribute)
            attrValues = dataset.get_distinct_valueset(attribute)
            # print(attrValues)
            if is_real_type and split_points > 0 and len(attrValues) > split_points:
                attrValues = sample(attrValues, split_points)
            for attrValue in attrValues:
                leftIdSet = []
                rightIdSet = []
                for Id in remainedSet:
                    instance = dataset.get_instance(Id)
                    value = instance[attribute]
                    # 将满足条件的放入左子树 什么条件？？？
                    if (is_real_type and value < attrValue)\
                            or(not is_real_type and value == attrValue):
                        leftIdSet.append(Id)
                    else:
                        rightIdSet.append(Id)
                leftTargets = [targets[id] for id in leftIdSet]
                rightTargets = [targets[id] for id in rightIdSet]
                sum_mse = MSE(leftTargets)+MSE(rightTargets)
                if mse < 0 or sum_mse < mse:
                    selectedAttribute = attribute
                    conditionValue = attrValue
                    mse = sum_mse
                    selectedLeftIdSet = leftIdSet
                    selectedRightIdSet = rightIdSet
        # print(selectedAttribute)
        # print(mse)
        # if not selectedAttribute or mse < 0:
        #     raise ValueError("cannot determine the split attribute.")
        tree = Tree()
        tree.split_feature = selectedAttribute
        # print(tree.split_feature)
        tree.real_value_feature = dataset.is_real_type_field(selectedAttribute)
        tree.conditionValue = conditionValue
        tree.leftTree, lf = construct_decision_tree(dataset, selectedLeftIdSet, targets, depth+1, lf, leaf_nodes, max_depth, loss)
        tree.rightTree, lf = construct_decision_tree(dataset, selectedRightIdSet, targets, depth+1, lf, leaf_nodes, max_depth, loss)
        return tree, lf
    else:  # 是叶子节点
        node = LeafNode(remainedSet)
        node.update_predict_value(targets, loss)
        leaf_nodes.append(node)
        lf.addnode(node)
        # tree.leaf_nodes.append(node)
        tree = Tree()
        tree.leafNode = node
        return tree, lf
