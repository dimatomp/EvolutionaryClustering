import math

import numpy as np

from .. import Constants
from .. import Metric
from .MabSolver import MabSolver
from ..RLthreadBase import ClusteringArmThread


class UCB(MabSolver):
    def __init__(self, action, is_fair=False, time_limit=None):
        MabSolver.__init__(self, action, time_limit)
        self.num = Constants.num_algos
        self.rewards = np.array([0.0] * self.num)
        # self.spendings = [[] for i in range(0, self.num)]
        # self.avg_spendings = [1] * Constants.num_algos
        self.n = np.array([1] * self.num)
        self.name = "ucb"
        self.iter = 1
        self.is_fair = is_fair

    # def initialize(self, f):
    #     print("\nInit UCB1")
    #     for i in range(0, self.num):
    #         ex = self.action  # AlgoExecutor
    #         t = ClusteringArmThread(ex.clu_algos[i], ex.metric, ex.X, ex.seed)
    #         random_cfg = t.clu_cs.sample_configuration()
    #
    #         # run on random config and get reward:
    #         reward = t.clu_run(random_cfg)
    #         self.rewards[i] = (Constants.in_reward - reward) / Constants.in_reward
    #
    #         if reward < ex.best_val:
    #             ex.best_val = reward
    #             ex.best_param = random_cfg
    #             ex.best_algo = ex.clu_algos[i]

    def initialize(self, f, true_labels=None):
        """
        Initialize rewards. We use here the same value,
        gained by calculating metrics on randomly assigned labels.
        """
        print("\nInit UCB1")
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
        values = self.rewards
        if self.is_fair:
            values = values / (self.sum_spendings / self.n)

        values = values + math.sqrt(2 * math.log(self.iter)) / self.n
        return np.argmax(values)

    def register_action(self, arm, time_consumed, reward):
        self.iter += 1
        self.rewards[arm] += reward
        self.n[arm] += 1
