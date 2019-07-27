from sys import float_info

import numpy as np

from . import Constants
from . import ThreadsRand as t


class AlgorithmExecutor:
    clu_algos = [Constants.kmeans_algo,
                 Constants.affinity_algo,
                 Constants.mean_shift_algo,
                 Constants.ward_algo,
                 Constants.dbscan_algo,
                 Constants.gm_algo,
                 Constants.bgm_algo
                 ]

    def __init__(self, num, metric, X, seed):
        self.metric = metric
        self.X = X
        self.run_num = np.array([0] * num)
        self.smacs = [0] * num
        self.best_val = Constants.best_init
        self.best_param = dict()
        self.best_algo = ""
        self.seed = seed

    def apply(self, arm, file, iteration_number):
        th = t.ClusteringRandThread(self.clu_algos[arm], self.metric, self.X, self.seed)
        th.run()
        #th.start()
        #th.join()
        self.run_num[arm] += 1
        reward = th.value
        if (reward < self.best_val):
            self.best_val = reward
            self.best_param = th.parameters
            self.best_algo = th.thread_name
        file.write(str(iteration_number) + ', ' + self.metric + ', ' + str(self.best_val) + ', ' + self.best_algo +
                   ', ' + self.clu_algos[arm] + ', ' + str(reward) + '\n')
        return (Constants.in_reward - reward) / Constants.in_reward
