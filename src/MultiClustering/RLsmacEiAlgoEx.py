import math
import sys

import numpy as np

from . import Constants
from . import RLthread as t


class RLsmacEiAlgoEx:
    clu_algos = Constants.algos

    def __init__(self, metric, X, seed, batch_size):
        self.metric = metric
        self.X = X
        self.run_num = np.array([0] * Constants.num_algos)
        self.best_val = Constants.best_init
        self.best_param = dict()
        self.best_algo = ""
        self.seed = seed
        self.batch_size = batch_size
        self.smacs = []
        self.th = []
        self.last_ei = [0.0] * Constants.num_algos
        self.tops_log = []

        # create all clustering threads in advance:
        for i in range(0, Constants.num_algos):
            self.th.append(t.RLthread(self.clu_algos[i], self.metric, self.X, self.seed, self.batch_size))
            self.smacs.append(self.th[i].smac)

    def apply(self, arm, file, iteration_number, remaining_time=None):
        th = self.th[arm]

        # initially, run_num for each arm == 0, thus we allocate 1 batch of target f calls:
        th.new_scenario(self.run_num[arm] + 1, remaining_time)  # add budget
        th.run()

        self.run_num[arm] += 1
        reward = th.value
        if reward < self.best_val:
            self.best_val = reward
            self.best_param = th.parameters
            self.best_algo = th.thread_name
        file.write(str(iteration_number) + ', ' + self.metric + ', ' + str(self.best_val) + ', ' + self.best_algo +
                   ', ' + self.clu_algos[arm] + ', ' + str(reward) + '\n')

        # as a reward return the (negative) Ei gain
        tops = []
        for i in range(0, len(self.smacs)):
            try:
                # will throw exception if smac wasn't run yet
                forest_tops = np.array(self.smacs[i].solver.model.rf.get_largest_values_of_trees())
            except:
                forest_tops = [0.0]

            tops.append(np.median(forest_tops))
        self.tops_log.append(tops)

        if not math.isfinite(tops[arm]):
            diff = 0  # protection from nan's and inf's
        else:
            diff = self.last_ei[arm] - tops[arm]   # old - new  (better to decrease EI)
            self.last_ei[arm] = tops[arm]

        # TODO or -diff? is it better to decrease ei or increase it?
        return -diff   # -diff is new-old
