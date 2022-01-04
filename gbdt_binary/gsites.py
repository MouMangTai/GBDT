from sklearn.ensemble import RandomForestClassifier
import operator

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from gutility import gen_atta_size, VotingClassifier
from gtask import Task
from InData.input_datas import IN_DataSet
import random

dropout_rate = 0.05


def generate_tasks(node_list, node_local, task_num, max_iter, sample_rate, learn_rate,
                   max_depth):
    base_minuend = node_list[0].id
    # print(base_minuend)
    # print(len(node_list))
    for node in node_list:
        node.task_list = []

        for i in range(1, task_num + 1):
            task_id = (node.id - base_minuend) * task_num + i
            _tree = None
            t = Task(
                task_id, node.id, _tree, max_iter, sample_rate, learn_rate,
                max_depth, split_points=0)

            temp_list = [t]
            node.task_list = node.task_list + temp_list
            node_local.task_list.append(t)

    return


# 第一轮第一项
def modified_local_je1(node_list, node_local, max_iter,
                       sample_rate, learn_rate, max_depth, loss, num_choice):
    print('label of Node:', len(node_list[0]
                                .fed_train.label))
    generate_tasks(node_list, node_local, 1, max_iter,
                   sample_rate, learn_rate, max_depth)
    f = dict()  # 记录F_{m-1}的值
    if num_choice == 0:
        return
    new_list = node_local.choice_set(num_choice)
    for node_index, node in enumerate(new_list):
        for task_index, task in enumerate(node.task_list):
            print("round: " + str(task_index) + " " + str(node_index))
            indices = np.random.permutation(len(node.origin_dataset))
            # node.origin_dataset = node.origin_dataset[indices]
            # print(node.origin_dataset)
            # node.fed_train = IN_DataSet(node.origin_dataset)
            print('The first round start')
            tup = task.gtask_ie1_je1(node.fed_train, f, loss, node_local, node)

            node_local.concatenate_interaction_model(tup.get_tree(), tup)  # true
            node_local.concatenate_model_local(tup.get_tree(), tup)  # true

    node_local.wrap_boosting()
    return f, node_list


# 第一轮非第一项
def modified_local_jl1(node_list, node_local, max_iter,
                       sample_rate, learn_rate, max_depth, f, loss, num_choice):
    if num_choice == 0:
        return
    new_list = node_local.choice_set(num_choice)
    # f = dict()
    for node_index, node in enumerate(new_list):
        # temp_M = node_concate(node, node_local.M_for_center, class_num, node_local)
        MM = node.temp_concatenate(node_local.M_for_center)
        MM_model = VotingClassifier(MM.get_trees())
        pre = MM_model.predict(node.fed_valid)
        # print(len(node.fed_valid.label))
        accuracy = accuracy_score(node.fed_valid.label, pre)
        node.trace_accuracy = np.concatenate([node.trace_accuracy, [accuracy]])
        node.cur_accuracy = accuracy
        errorr = 1 - accuracy
        node.trace_error = np.concatenate([node.trace_error, [errorr]])

        node.cur_error = errorr
        for task_index, task in enumerate(node.task_list):
            indices = np.random.permutation(len(node.origin_dataset))
            # node.origin_dataset = node.origin_dataset[indices]
            # node.fed_train = IN_DataSet(node.origin_dataset)
            tup = task.gtask_il1_je1(node_local.cur_treelfs, node.fed_train, f, loss, node_local, node)

            node_local.concatenate_interaction_model(tup.get_tree(), tup)
            node_local.concatenate_model_local(tup.get_tree(), tup)
    # node_list = dropout(node_list, dropout_rate)
    node_local.wrap_boosting()

    return node_list
