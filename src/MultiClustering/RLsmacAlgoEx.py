
import numpy as np

from . import Constants
from . import RLthread as t


class RLsmacAlgoEx:
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
        return (Constants.in_reward - reward) / Constants.in_reward
