import threading

import numpy as np
from ConfigSpace import InCondition
from ConfigSpace import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from smac.configspace import ConfigurationSpace
from .customsmac.smac_facade import SMAC
from smac.scenario.scenario import Scenario

from . import Constants
from . import Metric
from .RLthreadBase import ClusteringArmThread
from .customsmac.ei_optimization import InterleavedLocalAndRandomSearch
from .customsmac.smbo import SMBO


class RLthread(ClusteringArmThread):

    def __init__(self, name, metric, X, seed, batch_size):
        self.run_count = batch_size
        ClusteringArmThread.__init__(self, name, metric, X, seed)  # populates config space
        self.smac = None
        self.new_scenario(1)  # initial scenario

        # create smac with custom smbo:
        self.custom_acq_opt = InterleavedLocalAndRandomSearch(None, self.clu_cs, np.random.RandomState(self.seed))
        self.smac = SMAC(scenario=self.clu_scenario, rng=self.seed, tae_runner=self.clu_run, smbo_class=SMBO,
                         acquisition_function_optimizer=self.custom_acq_opt, expansion_number=5000)
        print(str(self.custom_acq_opt.acquisition_function))

    def new_scenario(self, c, remaining_time=None):
        # remaining_time is usually expected to be way more than needed for one call,
        # but sometimes it's guarding from hanging arbitraty long in single iteration
        if remaining_time is None:
            self.clu_scenario = Scenario({"run_obj": "quality",
                                          "cs": self.clu_cs,
                                          "deterministic": "true",
                                          "runcount-limit": self.run_count * c
                                          })
        else:
            self.clu_scenario = Scenario({"run_obj": "quality",
                                          "cs": self.clu_cs,
                                          "deterministic": "true",
                                          "tuner-timeout": remaining_time,
                                          "wallclock_limit": remaining_time,
                                          "cutoff_time": remaining_time,
                                          "runcount-limit": self.run_count * c
                                          })

        if hasattr(self.smac, "stats"):
            self.smac.stats._Stats__scenario = self.clu_scenario

    def run(self):
        print('Run SMAC ' + self.thread_name)
        # if self.state == Constants.run:
        #     # self.smac = SMAC(scenario=self.clu_scenario, rng=self.seed, tae_runner=self.clu_run, smbo_class=SMBO)
        #     # self.smac = SMAC(scenario=self.clu_scenario, rng=self.seed, tae_runner=self.clu_run)
        #     pass

        self.parameters = self.smac.optimize()  # aka Configuration
        self.value = self.smac.get_runhistory().get_cost(self.parameters)
