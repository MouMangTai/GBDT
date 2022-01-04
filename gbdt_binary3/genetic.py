# coding: utf-8
# https://github.com/zhaoxingfeng/Genetic-Algorithm/blob/master/GA/GA.py
from __future__ import division
import random
import math
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import pickle
import numpy as np
import pandas as pd
from gutility import VotingClassifier
from sklearn.metrics import accuracy_score


class GA(object):
    """
    f：目标函数
    lb：变量的最小取值
    ub：变量的最大取值
    maxiter：最大迭代次数
    sizepop：种群数量
    lenchrom：染色体长度
    pc：交叉概率
    pm：变异概率
    dim：变量的维度
    gamma: 权重阈值
    """
    def __init__(self, estimators, t_x, t_y, lb, ub, maxiter=10, sizepop=50, lenchrom=10, pc=0.8, pm=0.01, dim=10, gamma=0.05):
        self.maxiter = maxiter
        self.sizepop = sizepop
        self.lenchrom = lenchrom
        self.pc = pc
        self.pm = pm
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.t_x = t_x
        self.t_y = t_y
        self.gamma = gamma
        self.estimators = estimators
    
    def f(self, w):
        # 映射为sum为1的权重
        weights = [i / sum(w) for i in w]
        # 得到权重达到阈值的estimators
        current_estimators = [self.estimators[index] for index, weight in enumerate(weights) if weight >= self.gamma]
        # 重新得到分类器
        current_eclf = VotingClassifier(estimators=current_estimators)
        # 预测准确度
        score = accuracy_score(self.t_y, y_pred=current_eclf.predict(self.t_x))
        # 返回评分
        return 1 / ((1 - score) + 1e-6)
    
    def final_est(self, w):
        # 映射为sum为1的权重
        weights = [i / sum(w) for i in w]
        # 得到权重达到阈值的estimators
        current_estimators = [self.estimators[index] for index, weight in enumerate(weights) if weight >= self.gamma]
        return current_estimators
        
        
    # 初始化种群：返回一个三维数组，第一维是种子，第二维是变量维度，第三维是编码基因
    def Initialization(self):
        pop = []
        for i in range(self.sizepop):
            temp1 = []
            for j in range(self.dim):
                temp2 = []
                for k in range(self.lenchrom):
                    temp2.append(random.randint(0, 1))
                temp1.append(temp2)
            pop.append(temp1)
        return pop

    # 将二进制转化为十进制
    def b2d(self, pop_binary):
        pop_decimal = []
        for i in range(len(pop_binary)):
            temp1 = []
            for j in range(self.dim):
                temp2 = 0
                for k in range(self.lenchrom):
                    temp2 += pop_binary[i][j][k] * math.pow(2, k)
                temp2 = temp2 * (self.ub[j] - self.lb[j]) / (math.pow(2, self.lenchrom) - 1) + self.lb[j]
                temp1.append(temp2)
            pop_decimal.append(temp1)
        return pop_decimal

    # 轮盘赌模型选择适应值较高的种子
    def Roulette(self, fitness, pop):
        # 适应值按照大小排序
        sorted_index = np.argsort(fitness)
        sorted_fitness, sorted_pop = [], []
        for index in sorted_index:
            sorted_fitness.append(fitness[index])
            sorted_pop.append(pop[index])

        # 生成适应值累加序列
        fitness_sum = sum(sorted_fitness)
        accumulation = [None for col in range(len(sorted_fitness))]
        accumulation[0] = sorted_fitness[0] / fitness_sum
        for i in range(1, len(sorted_fitness)):
            accumulation[i] = accumulation[i - 1] + sorted_fitness[i] / fitness_sum

        # 轮盘赌
        roulette_index = []
        for j in range(len(sorted_fitness)):
            p = random.random()
            for k in range(len(accumulation)):
                if accumulation[k] >= p:
                    roulette_index.append(k)
                    break
        temp1, temp2 = [], []
        for index in roulette_index:
            temp1.append(sorted_fitness[index])
            temp2.append(sorted_pop[index])
        newpop = [[x, y] for x, y in zip(temp1, temp2)]
        newpop.sort()
        newpop_fitness = [newpop[i][0] for i in range(len(sorted_fitness))]
        newpop_pop = [newpop[i][1] for i in range(len(sorted_fitness))]
        return newpop_fitness, newpop_pop

    # 交叉繁殖：针对每一个种子，随机选取另一个种子与之交叉。
    # 随机取种子基因上的两个位置点，然后互换两点之间的部分
    def Crossover(self, pop):
        newpop = []
        for i in range(len(pop)):
            if random.random() < self.pc:
                # 选择另一个种子
                j = i
                while j == i:
                    j = random.randint(0, len(pop) - 1)
                cpoint1 = random.randint(1, self.lenchrom - 1)
                cpoint2 = cpoint1
                while cpoint2 == cpoint1:
                    cpoint2 = random.randint(1, self.lenchrom - 1)
                cpoint1, cpoint2 = min(cpoint1, cpoint2), max(cpoint1, cpoint2)
                newpop1, newpop2 = [], []
                for k in range(self.dim):
                    temp1, temp2 = [], []
                    temp1.extend(pop[i][k][0:cpoint1])
                    temp1.extend(pop[j][k][cpoint1:cpoint2])
                    temp1.extend(pop[i][k][cpoint2:])
                    temp2.extend(pop[j][k][0:cpoint1])
                    temp2.extend(pop[i][k][cpoint1:cpoint2])
                    temp2.extend(pop[j][k][cpoint2:])
                    newpop1.append(temp1)
                    newpop2.append(temp2)
                newpop.extend([newpop1, newpop2])
        return newpop

    # 变异：针对每一个种子的每一个维度，进行概率变异，变异基因为一位
    def Mutation(self, pop):
        newpop = copy.deepcopy(pop)
        for i in range(len(pop)):
            for j in range(self.dim):
                if random.random() < self.pm:
                    mpoint = random.randint(0, self.lenchrom - 1)
                    newpop[i][j][mpoint] = 1 - newpop[i][j][mpoint]
        return newpop

    # 绘制迭代-误差图
    def Ploterro(self, Convergence_curve):
        mpl.rcParams['font.sans-serif'] = ['Courier New']
        mpl.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=(10, 6))
        x = [i for i in range(len(Convergence_curve))]
        plt.plot(x, Convergence_curve, 'r-', linewidth=1.5, markersize=5)
        plt.xlabel(u'Iter', fontsize=18)
        plt.ylabel(u'Best score', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(0, )
        plt.grid(True)
        plt.show()

    def Run(self):
        pop = self.Initialization()
        errolist = []
        best_fitness = 0
        best_pos = []
        # print(self.maxiter)
        for Current_iter in range(self.maxiter):
            print("Iter = " + str(Current_iter))
            pop1 = self.Crossover(pop)
            pop2 = self.Mutation(pop1)
            pop3 = self.b2d(pop2)
            fitness = []
            for j in range(len(pop2)):
                # 每个个体的fitness
                fitness.append(self.f(pop3[j]))
            sorted_fitness, sorted_pop = self.Roulette(fitness, pop2)
            best_fitness = sorted_fitness[-1]
            best_pos = self.b2d([sorted_pop[-1]])[0]
            pop = sorted_pop[-1:-(self.sizepop + 1):-1]
            errolist.append(1 / best_fitness)
            if 1 / best_fitness < 0.0001:
                return best_fitness, best_pos, errolist
        return best_fitness, best_pos, errolist
