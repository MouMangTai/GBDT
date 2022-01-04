from sklearn.ensemble import RandomForestClassifier
import operator

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from gutility import gen_atta_size, VotingClassifier
from gtask import Task
from genetic import GA
import random

dropout_rate = 0.05


def f(w, gamma, estimators):
    # 映射为sum为1的权重
    weights = [i / sum(w) for i in w]
    # 得到权重达到阈值的estimators
    current_estimators = [estimators[index] for index, weight in enumerate(weights) if weight >= gamma]
    # 重新得到分类器
    current_eclf = VotingClassifier(
        estimators=current_estimators,
        weights=[1 / len(current_estimators) for i in range(len(current_estimators))],
        class_num=10)
    # 预测准确度
    score = accuracy_score(t_y, y_pred=current_eclf.predict(t_x))
    # 返回评分
    return 1 / ((1 - score) + 1e-6)


def generate_tasks(node_list, node_local, task_num, max_iter, sample_rate, learn_rate, max_depth):
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


def get_pruned(w, gamma, estimators, lflist):
    # 映射为sum为1的权重
    weights = [i / sum(w) for i in w]
    # 得到权重达到阈值的estimators
    current_estimators = [estimators[index] for index, weight in enumerate(weights) if weight >= gamma]
    current_lfs = [lflist[index] for index, weight in enumerate(weights) if weight >= gamma]

    return current_estimators, current_lfs


def gassen_distrib(node_list, ensemble, lflist, node_local, num_choice):
    if num_choice == 0:
        return
    new_list = node_local.choice_set(num_choice)
    num_est = len(ensemble)
    aver_weight = [0 for i in range(num_est)]
    for node_index, node in enumerate(new_list):
        valid_dataset = node.fed_train
        valid_label = node.fed_train.label
        a = GA(ensemble, node.fed_train, node.fed_train.label, [0 for i in range(num_est)], [1 for i in range(num_est)],
               10, 50, 14, 0.8, 0.01, num_est, 0.05)
        best_score, best_pos, error_list = a.Run()
        print("Best Predict Score = " + str(round(1 - (1 / (best_score - 1e-6)), 4)))
        best_weights = [i / sum(best_pos) for i in best_pos]
        for i in range(num_est):
            aver_weight[i] += best_weights[i]
        # print("Best Weights = " + str([round(a, 4) for a in best_weights]))
    aver_weight[i] = aver_weight[i] / num_choice
    gamma = 0.05
    estimators_index = [index for index, weight in enumerate(aver_weight) if weight >= 0.05]
    pruned_est, est_lfs = get_pruned(aver_weight, gamma, ensemble, lflist)
    # print(type(est_lfs[0]))
    return pruned_est, est_lfs


def sequential_local_je1(is_selection, node_list, node_local, max_iter,
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
            if node_index == 0:
                tup = task.gtask_ie1_je1(node.fed_train, f, loss, node_list, node)
            else:
                tup = task.gtask_il1_je1(node_local.cur_treelfs, node.fed_train, f, loss, node_list, node)

            node_local.concatenate_temp(tup.get_tree(), tup)
            node_local.concatenate_model_local(tup.get_tree(), tup)

    n_choice = 10
    # print(type(node_local.temp_treelfs[0]))

    node_local.clean_base_weight_temp()
    node_local.wrap_boosting()
    return f, node_list


def modified_local_je1(is_selection, node_list, node_local, max_iter,
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
            tup = task.gtask_ie1_je1(node.fed_train, f, loss, node_list, node)
            node_local.concatenate_temp(tup.get_tree(), tup)

    n_choice = 10
    # print(type(node_local.temp_treelfs[0]))
    if is_selection:
        tem_trees, tem_treelfs = gassen_distrib(node_list, node_local.temp_trees, node_local.temp_treelfs, node_local,
                                                n_choice)
    else:
        tem_trees = node_local.temp_trees
        tem_treelfs = node_local.temp_treelfs
    print(len(tem_trees))
    node_local.batch_concatenate_model_local(tem_trees, tem_treelfs)
    node_local.clean_base_weight_temp()
    node_local.wrap_boosting()
    return f, node_list


def modified_local_jl1(is_selection, node_list, node_local, max_iter,
                       sample_rate, learn_rate, max_depth, f, loss, num_choice):
    if num_choice == 0:
        return
    new_list = node_local.choice_set(num_choice)
    f = dict()
    for node_index, node in enumerate(new_list):
        # temp_M = node_concate(node, node_local.M_for_center, class_num, node_local)
        # print(len(node_local.M_for_center.get_trees()))
        MM_model = VotingClassifier(node_local.M_for_center.get_trees())

        # MM_model = VotingClassifier(MM.get_trees())
        pre = MM_model.predict(node.fed_valid)
        # print(len(node.fed_valid.label))
        accuracy = accuracy_score(node.fed_valid.label, pre)
        # print(node.fed_valid.label)
        # print(pre)
        print(accuracy)
        node.trace_accuracy = np.concatenate([node.trace_accuracy, [accuracy]])
        node.cur_accuracy = accuracy
        errorr = 1 - accuracy
        node.trace_error = np.concatenate([node.trace_error, [errorr]])

        node.cur_error = errorr
        for task_index, task in enumerate(node.task_list):
            indices = np.random.permutation(len(node.origin_dataset))

            tup = task.gtask_il1_je1(node_local.cur_treelfs, node.fed_train, f, loss, node_list, node)
            node_local.concatenate_temp(tup.get_tree(), tup)

    n_choice = 10
    if is_selection:
        tem_trees, tem_treelfs = gassen_distrib(node_list, node_local.temp_trees, node_local.temp_treelfs, node_local,
                                                n_choice)
    else:
        tem_trees = node_local.temp_trees
        tem_treelfs = node_local.temp_treelfs
    print(len(tem_trees))
    node_local.batch_concatenate_model_local(tem_trees, tem_treelfs)

    node_local.clean_base_weight_temp()
    node_local.wrap_boosting()

    return node_list


def sequential_local_jl1(is_selection, node_list, node_local, max_iter,
                         sample_rate, learn_rate, max_depth, f, loss, num_choice):
    if num_choice == 0:
        return
    new_list = node_local.choice_set(num_choice)
    f = dict()
    for node_index, node in enumerate(new_list):
        # temp_M = node_concate(node, node_local.M_for_center, class_num, node_local)
        # print(len(node_local.M_for_center.get_trees()))
        MM_model = VotingClassifier(node_local.M_for_center.get_trees())

        # MM_model = VotingClassifier(MM.get_trees())
        pre = MM_model.predict(node.fed_valid)
        # print(len(node.fed_valid.label))
        accuracy = accuracy_score(node.fed_valid.label, pre)
        # print(node.fed_valid.label)
        # print(pre)
        print(accuracy)
        node.trace_accuracy = np.concatenate([node.trace_accuracy, [accuracy]])
        node.cur_accuracy = accuracy
        errorr = 1 - accuracy
        node.trace_error = np.concatenate([node.trace_error, [errorr]])

        node.cur_error = errorr
        for task_index, task in enumerate(node.task_list):
            indices = np.random.permutation(len(node.origin_dataset))

            tup = task.gtask_il1_je1(node_local.cur_treelfs, node.fed_train, f, loss, node_list, node)
            node_local.concatenate_temp(tup.get_tree(), tup)
            node_local.concatenate_model_local(tup.get_tree(), tup)
    n_choice = 10

    node_local.clean_base_weight_temp()
    node_local.wrap_boosting()

    return node_list
