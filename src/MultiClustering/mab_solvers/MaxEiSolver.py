import numpy as np
from numpy.random import choice

from .. import Constants
from .MabSolver import MabSolver


class MaxEi(MabSolver):
    def __init__(self, action, optimizers, time_limit=None):
        MabSolver.__init__(self, action, time_limit)
        # self.sum_spendings = [1] * Constants.num_algos
        self.num = Constants.num_algos
        self.rewards = np.array([0.0] * self.num)
        # self.spendings = [[] for i in range(0, self.num)]
        self.n = np.array([0] * self.num)
        self.optimizers = optimizers
        self.name = "MaxEi"
        self.iter = 1

        # initially each arm appears here.
        # if not empty, arms should be drawn from here.
        #   otherwise - according to max-ei algo
        self.pending = [i for i in range(0, Constants.num_algos)]
        self.tops_log = []

    def initialize(self, f, true_labels=None):
        print("\nInit MaxEi")
        self.pending = [i for i in range(0, Constants.num_algos)]

    def draw(self):
        if len(self.pending) != 0:
            arm = self.pending[-1]
            self.pending = self.pending[:-1]
            self.tops_log.append([])
            return arm

        tops = []
        for i in range(0, len(self.optimizers)):
            tops.append(self.action.th[i].optimizer.get_best_from_forest())

        self.tops_log.append(tops)

        # The actual formula here:
        # spent_by_arm = np.array([np.sum(x) for x in self.spendings])
        # # spent_by_arm = spent_by_arm + 1   # prevent log becoming negative
        # tops = np.array(tops)
        #
        # log_time = math.log(np.sum(spent_by_arm))
        # rate = max(log_time, 1) / spent_by_arm
        # tops = np.add(tops, np.sqrt(rate * 2))
        # return np.argmax(tops)

        x = np.array(tops, dtype=np.double)
        x = x / np.linalg.norm(x)
        e_x = np.exp(x - np.max(x))
        s_max = e_x / e_x.sum(axis=0)

        d = choice([i for i in range(0, Constants.num_algos)], p=s_max)
        return d


    def register_action(self, arm, time_consumed, reward):
        self.iter += 1
        self.rewards[arm] += reward
        self.n[arm] += 1
        # self.avg_spendings[arm] = (self.avg_spendings[arm] * self.n[arm] + time_consumed) / (self.n[arm] + 1.0)
        # self.spendings[arm].append(time_consumed)
