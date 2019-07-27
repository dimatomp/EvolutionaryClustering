from sys import float_info

import numpy as np

from . import Constants
from . import ExThreads as t


class EXsmacAlgoExecutor:
    def __init__(self, num, metric, X, seed, limit_type):
        self.metric = metric
        self.X = X
        self.run_num = np.array([0] * num)
        self.smacs = []
        self.best_val = Constants.best_init
        self.best_param = dict()
        self.best_algo = "-1"
        self.seed = seed
        self.saved_parameters = dict()
        self.num_algos = num
        self.limit_type = limit_type
        self.per_algo = {}

    def apply(self, budget):
        # just shortcuts
        metric = self.metric
        X = self.X
        seed = self.seed
        # saved_parameters = self.saved_parameters

        best_algo = "-1"
        best_params = dict()
        best_val = Constants.best_init

        th = [0] * 7
        th[0] = t.km_thread(metric, X, seed, int(budget / self.num_algos), self.limit_type)
        th[1] = t.aff_thread(metric, X, seed, int(budget / self.num_algos), self.limit_type)
        th[2] = t.ms_thread(metric, X, seed, int(budget / self.num_algos), self.limit_type)
        th[3] = t.w_thread(metric, X, seed, int(budget / self.num_algos), self.limit_type)
        th[4] = t.db_thread(metric, X, seed, int(budget / self.num_algos), self.limit_type)
        th[5] = t.gm_thread(metric, X, seed, int(budget / self.num_algos), self.limit_type)
        th[6] = t.bgm_thread(metric, X, seed, int(budget / self.num_algos), self.limit_type)

        per_algo = {}
        for a in Constants.algos:
            per_algo[a] = float_info.max

        for i in range(0, Constants.num_algos):
            # th[i].start()
            # th[i].join()
            th[i].run()

            self.smacs.append(th[i].smac)

            print(('\t\t\t\tFor algo ' + th[i].thread_name + ' with metric ' + metric +
                   ' lowest function value found: %f' % th[i].value))
            print(('\t\t\t\tParameter setting %s' % th[i].parameters))

            per_algo[th[i].thread_name] = min(per_algo[th[i].thread_name], th[i].value)

            if th[i].value < best_val:
                best_val = th[i].value
                best_algo = th[i].thread_name
                best_params = th[i].parameters

            # saved_parameters[metric][th[i].thread_name] = th[i].parameters

        reward = best_val
        if (reward < self.best_val):
            self.best_val = reward
            self.best_param = best_params
            self.best_algo = best_algo

        self.per_algo = per_algo
        return reward
