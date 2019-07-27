import sys

import numpy as np
from numpy.random import choice

from .. import Constants
from .. import Metric
from .MabSolver import MabSolver
from ..RLthreadBase import ClusteringArmThread
from .Softmax import Softmax


class SoftmaxR(Softmax):
    def __init__(self, action, tau=Constants.tau, is_fair=False, time_limit=None):
        super().__init__(action, tau, is_fair, time_limit)
        self.raw_rewards = []

    def initialize(self, f, true_labels=None):
        super().initialize(f)
        self.raw_rewards = np.array(self.rewards)

    def register_action(self, arm, time_consumed, reward):
        self.n[arm] += 1
        self.raw_rewards[arm] = reward
        self.rewards[arm] = reward
