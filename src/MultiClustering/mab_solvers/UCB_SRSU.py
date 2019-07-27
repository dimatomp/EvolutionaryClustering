import math

import numpy as np

from .. import Constants
from .. import Metric
from .MabSolver import MabSolver
from ..RLthreadBase import ClusteringArmThread
from .Softmax import Softmax
from .UCB import UCB

s_norm = Softmax.softmax_normalize


class UCBsrsu(UCB):
    def __init__(self, action, is_fair=False, time_limit=None):
        super().__init__(action, is_fair, time_limit)
        self.raw_rewards = []

    def initialize(self, f, true_labels=None):
        super().initialize(f)
        self.raw_rewards = np.array(self.rewards)

    def register_action(self, arm, time_consumed, reward):
        self.iter += 1
        self.n[arm] += 1
        self.raw_rewards[arm] = reward
        self.rewards = s_norm(self.raw_rewards) + s_norm(self.u_correction(self.sum_spendings))

