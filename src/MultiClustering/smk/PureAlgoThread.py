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

from .. import Constants
from .. import Metric


class ClusteringArmThread:
    def __init__(self, name, metric, X):
        self.thread_name = name
        self.metric = metric
        self.X = X
        self.clu_cs = ConfigurationSpace()

        if (name == Constants.kmeans_algo):
            algorithm = CategoricalHyperparameter("algorithm", ["auto", "full", "elkan"])
            tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2)
            n_clusters = UniformIntegerHyperparameter("n_clusters", 2, 15)
            n_init = UniformIntegerHyperparameter("n_init", 2, 15)
            max_iter = UniformIntegerHyperparameter("max_iter", 50, 1500)
            verbose = Constant("verbose", 0)
            self.clu_cs.add_hyperparameters([n_clusters, n_init, max_iter, tol, verbose, algorithm])

        elif (name == Constants.affinity_algo):
            damping = UniformFloatHyperparameter("damping", 0.5, 1.0)
            max_iter = UniformIntegerHyperparameter("max_iter", 100, 1000)
            convergence_iter = UniformIntegerHyperparameter("convergence_iter", 5, 20)
            self.clu_cs.add_hyperparameters([damping, max_iter, convergence_iter])

        elif (name == Constants.mean_shift_algo):
            quantile = UniformFloatHyperparameter("quantile", 0.0, 1.0)
            bin_seeding = UniformIntegerHyperparameter("bin_seeding", 0, 1)
            min_bin_freq = UniformIntegerHyperparameter("min_bin_freq", 1, 100)
            cluster_all = UniformIntegerHyperparameter("cluster_all", 0, 1)
            self.clu_cs.add_hyperparameters([quantile, bin_seeding, min_bin_freq, cluster_all])

        elif (name == Constants.ward_algo):
            linkage = CategoricalHyperparameter("linkage", ["ward", "complete", "average"])
            affinity_all = CategoricalHyperparameter("affinity_a",
                                                     ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"])
            affinity_ward = CategoricalHyperparameter("affinity_w", ["euclidean"])
            n_clusters = UniformIntegerHyperparameter("n_clusters", 2, 15)
            self.clu_cs.add_hyperparameters([n_clusters, affinity_all, affinity_ward, linkage])
            self.clu_cs.add_condition(InCondition(child=affinity_ward, parent=linkage, values=["ward"]))
            self.clu_cs.add_condition(
                InCondition(child=affinity_all, parent=linkage, values=["ward", "complete", "average"]))

        elif (name == Constants.dbscan_algo):
            algorithm = CategoricalHyperparameter("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
            eps = UniformFloatHyperparameter("eps", 0.1, 0.9)
            min_samples = UniformIntegerHyperparameter("min_samples", 2, 10)
            leaf_size = UniformIntegerHyperparameter("leaf_size", 5, 100)
            self.clu_cs.add_hyperparameters([eps, min_samples, algorithm, leaf_size])

        elif (name == Constants.gm_algo):
            cov_t = CategoricalHyperparameter("covariance_type", ["full", "tied", "diag", "spherical"])
            tol = UniformFloatHyperparameter("tol", 1e-6, 0.1)
            reg_c = UniformFloatHyperparameter("reg_covar", 1e-10, 0.1)
            n_com = UniformIntegerHyperparameter("n_components", 2, 15)
            max_iter = UniformIntegerHyperparameter("max_iter", 10, 1000)
            self.clu_cs.add_hyperparameters([cov_t, tol, reg_c, n_com, max_iter])

        elif (name == Constants.bgm_algo):
            cov_t = CategoricalHyperparameter("covariance_type", ["full", "tied", "diag", "spherical"])
            tol = UniformFloatHyperparameter("tol", 1e-6, 0.1)
            reg_c = UniformFloatHyperparameter("reg_covar", 1e-10, 0.1)
            wcp = UniformFloatHyperparameter("weight_concentration_prior", 1e-10, 0.1)
            mpp = UniformFloatHyperparameter("mean_precision_prior", 1e-10, 0.1)
            n_com = UniformIntegerHyperparameter("n_components", 2, 15)
            max_iter = UniformIntegerHyperparameter("max_iter", 10, 1000)
            self.clu_cs.add_hyperparameters([wcp, mpp, cov_t, tol, reg_c, n_com, max_iter])


    def clu_run(self, cfg):
        cl = None
        if (self.thread_name == Constants.kmeans_algo):
            cl = KMeans(**cfg)
        elif (self.thread_name == Constants.affinity_algo):
            cl = AffinityPropagation(**cfg)
        elif (self.thread_name == Constants.mean_shift_algo):
            bandwidth = estimate_bandwidth(self.X, quantile=cfg['quantile'])
            cl = MeanShift(bandwidth=bandwidth, bin_seeding=bool(cfg['bin_seeding']), min_bin_freq=cfg['min_bin_freq'],
                           cluster_all=bool(cfg['cluster_all']))
        elif (self.thread_name == Constants.ward_algo):
            linkage = cfg["linkage"]
            aff = ""
            if ("ward" in linkage):
                aff = cfg["affinity_w"]
            else:
                aff = cfg["affinity_a"]
            n_c = cfg["n_clusters"]
            cl = AgglomerativeClustering(n_clusters=n_c, linkage=linkage, affinity=aff)
        elif (self.thread_name == Constants.dbscan_algo):
            cl = DBSCAN(**cfg)
        elif (self.thread_name == Constants.gm_algo):
            cl = GaussianMixture(**cfg)
        elif (self.thread_name == Constants.bgm_algo):
            cl = BayesianGaussianMixture(**cfg)

        cl.fit(self.X)

        if (self.thread_name == Constants.gm_algo) or (self.thread_name == Constants.bgm_algo):
            labels = cl.predict(self.X)
        else:
            labels = cl.labels_

        # labels_unique = np.unique(labels)
        # n_clusters = len(labels_unique)
        value = self.metric(self.X, labels)
        # print(self.thread_name + " n_clusters=" + str(n_clusters) + " metric=" + str(value))
        return value