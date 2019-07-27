import abc
import threading

import numpy as np
from ConfigSpace import InCondition
from ConfigSpace import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from smac.configspace import ConfigurationSpace
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from enum import Enum
# from smac.optimizer.ei_optimization import InterleavedLocalAndEvolutionarySearch

# from smac.optimizer.ei_optimization import EASearch
# from smac.optimizer.ei_optimization import InterleavedEvolutionarySearch

from . import Constants
from . import Metric


class LimitType(Enum):
    TIME = 1
    CALLS = 2

class ClusteringThread(threading.Thread):
    def __init__(self, metric, X, seed, budget, limit_type: LimitType):
        threading.Thread.__init__(self)
        self.metric = metric
        self.X = X
        self.thread_name = None
        self.seed = seed
        self.value = Constants.bad_cluster
        self.parameters = dict()
        self.config_space = ConfigurationSpace()
        self.scenario = None
        self.rng = np.random.RandomState(seed)
        self.smac = None
        self.scenario_parameters = {"run_obj": "quality",
                                    "deterministic": "true"}

        if limit_type == LimitType.CALLS:
            self.limit_by_calls(budget)
        else:
            self.limit_by_time(budget)

        self.acq_optimizer = None
        # self.custom = InterleavedEvolutionarySearch(None, self.config_space, np.random.RandomState(self.seed))
        # self.custom = EASearch(None, self.config_space, max_generations=200, generation_size=20)

    # Adds scenario parameters to limit by CALLS budget
    def limit_by_calls(self, budget):
        # self.scenario_parameters["runcount-limit"] = budget
        self.scenario_parameters = {"run_obj": "quality",
                                    "runcount-limit": budget,
                                    "deterministic": "true"}

    # Adds scenario parameters to limit by TIME budget
    def limit_by_time(self, budget):
        # TODO maybe it's enough to limit one thing, not 3.
        self.scenario_parameters = {"run_obj": "quality",
                                    "tuner-timeout": budget,
                                    "wallclock_limit": budget,
                                    "cutoff_time": budget,
                                    "deterministic": "true"}

    def evaluate(self, labels):
        labels_unique = np.unique(labels)
        n_clusters = len(labels_unique)
        m = Metric.metric(self.X, n_clusters, labels, self.metric)
        return m

    @abc.abstractmethod
    def target_run(self, cfg):
        raise NotImplementedError()

    # performs one step of optimization
    def run(self):
        print('Run ' + self.thread_name)
        smac = SMAC(scenario=self.scenario, rng=self.rng, tae_runner=self.target_run,
                    acquisition_function_optimizer=self.acq_optimizer)
        self.parameters = smac.optimize()
        self.value = smac.get_runhistory().get_cost(self.parameters)
        self.smac = smac  # expose smac to get some stats after the run


class km_thread(ClusteringThread):
    def __init__(self, metric, X, seed, budget, limit_type: LimitType):
        ClusteringThread.__init__(self, metric, X, seed, budget, limit_type)
        self.thread_name = Constants.kmeans_algo

        # configuration space
        algorithm = CategoricalHyperparameter("algorithm", ["auto", "full", "elkan"])
        tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2)
        n_clusters = UniformIntegerHyperparameter("n_clusters", 2, 15)
        n_init = UniformIntegerHyperparameter("n_init", 2, 15)
        max_iter = UniformIntegerHyperparameter("max_iter", 50, 1500)
        verbose = Constant(name="verbose", value=0)
        self.config_space.add_hyperparameters([n_clusters, n_init, max_iter, tol, algorithm])

        self.scenario_parameters["cs"] = self.config_space
        self.scenario = Scenario(self.scenario_parameters)

    def target_run(self, cfg):
        cl = KMeans(**cfg)
        labels = cl.fit_predict(self.X)
        return self.evaluate(labels)


class aff_thread(ClusteringThread):
    def __init__(self, metric, X, seed, budget, limit_type: LimitType):
        ClusteringThread.__init__(self, metric, X, seed, budget, limit_type)
        self.thread_name = Constants.affinity_algo

        damping = UniformFloatHyperparameter("damping", 0.5, 1.0)
        max_iter = UniformIntegerHyperparameter("max_iter", 100, 1000)
        convergence_iter = UniformIntegerHyperparameter("convergence_iter", 5, 20)
        verbose = Constant("verbose", 0)

        self.config_space.add_hyperparameters([damping, max_iter, convergence_iter])
        self.scenario_parameters["cs"] = self.config_space
        self.scenario = Scenario(self.scenario_parameters)

    def target_run(self, cfg):
        cl = AffinityPropagation(**cfg)
        labels = cl.fit_predict(self.X)
        return self.evaluate(labels)


class ms_thread(ClusteringThread):
    def __init__(self, metric, X, seed, budget, limit_type: LimitType):
        ClusteringThread.__init__(self, metric, X, seed, budget, limit_type)
        self.thread_name = Constants.mean_shift_algo

        quantile = UniformFloatHyperparameter("quantile", 0.0, 1.0)
        bin_seeding = UniformIntegerHyperparameter("bin_seeding", 0, 1)
        min_bin_freq = UniformIntegerHyperparameter("min_bin_freq", 1, 100)
        cluster_all = UniformIntegerHyperparameter("cluster_all", 0, 1)
        verbose = Constant("verbose", 0)
        self.config_space.add_hyperparameters([quantile, bin_seeding, min_bin_freq, cluster_all])

        self.scenario_parameters["cs"] = self.config_space
        self.scenario = Scenario(self.scenario_parameters)

    def target_run(self, cfg):
        bandwidth = estimate_bandwidth(self.X, quantile=cfg['quantile'])
        cl = MeanShift(bandwidth=bandwidth, bin_seeding=bool(cfg['bin_seeding']), min_bin_freq=cfg['min_bin_freq'],
                       cluster_all=bool(cfg['cluster_all']))
        labels = cl.fit_predict(self.X)
        return self.evaluate(labels)


class w_thread(ClusteringThread):
    def __init__(self, metric, X, seed, budget, limit_type: LimitType):
        ClusteringThread.__init__(self, metric, X, seed, budget, limit_type)
        self.thread_name = Constants.ward_algo

        linkage = CategoricalHyperparameter("linkage", ["ward", "complete", "average"])
        affinity_all = CategoricalHyperparameter("affinity_a",
                                                 ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"])
        affinity_ward = CategoricalHyperparameter("affinity_w", ["euclidean"])
        n_clusters = UniformIntegerHyperparameter("n_clusters", 2, 15)
        verbose = Constant("verbose", 0)

        self.config_space.add_hyperparameters([n_clusters, affinity_all, affinity_ward, linkage])
        self.config_space.add_condition(InCondition(child=affinity_ward, parent=linkage, values=["ward"]))
        self.config_space.add_condition(
            InCondition(child=affinity_all, parent=linkage, values=["ward", "complete", "average"]))

        self.scenario_parameters["cs"] = self.config_space
        self.scenario = Scenario(self.scenario_parameters)

    def target_run(self, cfg):
        linkage = cfg["linkage"]
        aff = ""
        if ("ward" in linkage):
            aff = cfg["affinity_w"]
        else:
            aff = cfg["affinity_a"]
        n_c = cfg["n_clusters"]
        cl = AgglomerativeClustering(n_clusters=n_c, linkage=linkage, affinity=aff)
        labels = cl.fit_predict(self.X)
        return self.evaluate(labels)


class db_thread(ClusteringThread):
    def __init__(self, metric, X, seed, budget, limit_type: LimitType):
        ClusteringThread.__init__(self, metric, X, seed, budget, limit_type)
        self.thread_name = Constants.dbscan_algo

        algorithm = CategoricalHyperparameter("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
        eps = UniformFloatHyperparameter("eps", 0.1, 0.9)
        min_samples = UniformIntegerHyperparameter("min_samples", 2, 10)
        leaf_size = UniformIntegerHyperparameter("leaf_size", 5, 100)
        self.config_space.add_hyperparameters([eps, min_samples, algorithm, leaf_size])

        self.scenario_parameters["cs"] = self.config_space
        self.scenario = Scenario(self.scenario_parameters)

    def target_run(self, cfg):
        cl = DBSCAN(**cfg)
        labels = cl.fit_predict(self.X)
        return self.evaluate(labels)


class gm_thread(ClusteringThread):
    def __init__(self, metric, X, seed, budget, limit_type: LimitType):
        ClusteringThread.__init__(self, metric, X, seed, budget, limit_type)
        self.thread_name = Constants.gm_algo

        cov_t = CategoricalHyperparameter("covariance_type", ["full", "tied", "diag", "spherical"])
        tol = UniformFloatHyperparameter("tol", 1e-6, 0.1)
        reg_c = UniformFloatHyperparameter("reg_covar", 1e-10, 0.1)
        n_com = UniformIntegerHyperparameter("n_components", 2, 15)
        max_iter = UniformIntegerHyperparameter("max_iter", 10, 1000)
        self.config_space.add_hyperparameters([cov_t, tol, reg_c, n_com, max_iter])

        self.scenario_parameters["cs"] = self.config_space
        self.scenario = Scenario(self.scenario_parameters)

    def target_run(self, cfg):
        cl = GaussianMixture(**cfg)
        cl.fit(self.X)
        labels = cl.predict(self.X)
        return self.evaluate(labels)


class bgm_thread(ClusteringThread):
    def __init__(self, metric, X, seed, budget, limit_type: LimitType):
        ClusteringThread.__init__(self, metric, X, seed, budget, limit_type)
        self.thread_name = Constants.bgm_algo

        cov_t = CategoricalHyperparameter("covariance_type", ["full", "tied", "diag", "spherical"])
        tol = UniformFloatHyperparameter("tol", 1e-6, 0.1)
        reg_c = UniformFloatHyperparameter("reg_covar", 1e-10, 0.1)
        wcp = UniformFloatHyperparameter("weight_concentration_prior", 1e-10, 0.1)
        mpp = UniformFloatHyperparameter("mean_precision_prior", 1e-10, 0.1)
        n_com = UniformIntegerHyperparameter("n_components", 2, 15)
        max_iter = UniformIntegerHyperparameter("max_iter", 10, 1000)
        self.config_space.add_hyperparameters([wcp, mpp, cov_t, tol, reg_c, n_com, max_iter])

        self.scenario_parameters["cs"] = self.config_space
        self.scenario = Scenario(self.scenario_parameters)

    def target_run(self, cfg):
        cl = BayesianGaussianMixture(**cfg)
        cl.fit(self.X)
        labels = cl.predict(self.X)
        return self.evaluate(labels)

