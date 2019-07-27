import math
import random
from random import choice

import numpy as np

from .. import Constants
from .. import Metric
from .MabSolver import MabSolver
from ..RLthreadBase import ClusteringArmThread


class Uniform(MabSolver):
    def __init__(self, action, time_limit=None):
        MabSolver.__init__(self, action, time_limit)
        self.num = Constants.num_algos
        self.rewards = np.array([0.0] * self.num)
        self.n = np.array([1] * self.num)
        self.name = "ucb"
        self.iter = 1

    def initialize(self, f, true_labels=None):
        """
        Initialize rewards. We use here the same value,
        gained by calculating metrics on randomly assigned labels.
        """
        print("\nInit Uniform")
        n_clusters = 15
        labels = np.random.randint(0, n_clusters, len(self.action.X))
        for c in range(0, n_clusters):
            labels[c] = c
        np.random.shuffle(labels)
        res = Metric.metric(self.action.X, n_clusters, labels, self.action.metric, true_labels)

        # start = time.time()
        for i in range(0, Constants.num_algos):
            self.rewards[i] = -res  # the smallest value is, the better.
        # self.consume_limit(time.time() - start)
        f.write("Init rewards: " + str(self.rewards) + '\n')

    def draw(self):
        return choice([i for i in range(0, Constants.num_algos)])


    def register_action(self, arm, time_consumed, reward):
        self.iter += 1
        self.rewards[arm] += reward
        self.n[arm] += 1
