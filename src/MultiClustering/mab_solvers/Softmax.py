import numpy as np
from numpy.random import choice

from .. import Constants
from .. import Metric
from .MabSolver import MabSolver
from ..RLthreadBase import ClusteringArmThread


class Softmax(MabSolver):
    def __init__(self, action, tau, is_fair=False, time_limit=None):
        MabSolver.__init__(self, action, time_limit)
        num = Constants.num_algos
        self.rewards = np.array([0.0] * num)
        self.n = np.array([0] * num)
        self.tau = tau
        self.is_fair = is_fair
        # self.name = "softmax" + str(tau * 10)

    # def initialize(self, f):
    #     print("\nInit Softmax with tau = " + str(self.tau))
    #     # start = time.time()
    #     for i in range(0, Constants.num_algos):
    #         # self.r[i] = self.action.apply(i, f, i)
    #         # instead af calling smac ^ get some randon config and calculate reward for i-th algo:
    #
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
    #     # To make comparison more fair, we do not consume time for initialization
    #     # Because no actual clustering is involved, just random values
    #     # self.consume_limit(time.time() - start)

    def initialize(self, f, true_labels=None):
        """
        Initialize rewards. We use here the same value,
        gained by calculating metrics on randomly assigned labels.
        """
        print("\nInit Softmax with tau = " + str(self.tau))
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
        if not self.is_fair:
            s_max = self.softmax_normalize(self.rewards)
        else:
            x = np.array(self.rewards)
            x = x / (self.sum_spendings / self.n)
            s_max = self.softmax_normalize(x)

        d = choice([i for i in range(0, Constants.num_algos)], p=s_max)
        return d

    def register_action(self, arm, time_consumed, reward):
        self.rewards[arm] += reward
        self.n[arm] += 1

    @staticmethod
    def softmax_normalize(rewards):
        x = rewards
        x = x / np.linalg.norm(x)
        e_x = np.exp(x - np.max(x))
        s_max = e_x / e_x.sum(axis=0)
        return s_max
