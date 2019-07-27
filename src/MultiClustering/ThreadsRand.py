import threading
from . import Constants
from . import Metric

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as u
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

class ClusteringRandThread(threading.Thread):

    def __init__(self, name, metric, X, seed):
        threading.Thread.__init__(self)

        self.thread_name = name
        self.metric = metric
        self.X = X
        self.value = Constants.bad_cluster
        self.parameters = dict()
        self.seed = seed
        self.clu_cs = dict()
        #np.random.set_state(seed)

        if (name == Constants.kmeans_algo):
            a = 1e-6
            b = 1e-2
            self.clu_cs = {"algorithm": ["auto", "full", "elkan"],
                           "n_clusters": sp_randint(2, 15),
                           "n_init": sp_randint(2, 15),
                           "max_iter": sp_randint(50, 1500),
                           "verbose": sp_randint(0, 10),
                           "tol": (b - a) * np.random.random_sample() + a}

        elif (name == Constants.affinity_algo):
            a = 0.5
            b = 1.0
            self.clu_cs = {"damping": (b - a) * np.random.random_sample() + a,
                           "max_iter": sp_randint(100, 1000),
                           "convergence_iter": sp_randint(5, 20)}

        elif (name == Constants.mean_shift_algo):
            self.clu_cs = {"bin_seeding": [True, False],
                           "min_bin_freq": sp_randint(1, 100),
                           "cluster_all": [True, False]}

        elif (name == Constants.ward_algo):
            self.clu_cs = {"linkage": ["ward", "complete", "average"],
                           "affinity_a": ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"],
                           "affinity_w": ["euclidean"],
                           "n_clusters": sp_randint(2, 15)
            }

        elif (name == Constants.dbscan_algo):
            a = 0.1
            b = 0.9
            self.clu_cs = {"eps": (b - a) * np.random.random_sample() + a,
                           "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                           "min_samples": sp_randint(2, 10),
                           "leaf_size": sp_randint(5, 100)}

        elif (name == Constants.gm_algo):
            a = 1e-6
            b = 0.1
            c = 1e-10
            d = 0.1
            self.clu_cs = {"covariance_type": ["full", "tied", "diag", "spherical"],
                           "tol": (b - a) * np.random.random_sample() + a,
                           "reg_covar": (d - c) * np.random.random_sample() + c,
                           "n_components": sp_randint(2, 15),
                           "max_iter": sp_randint(10, 1000)}

        elif (name == Constants.bgm_algo):
            a = 1e-6
            b = 0.1
            c = 1e-10
            d = 0.1
            self.clu_cs = {"covariance_type": ["full", "tied", "diag", "spherical"],
                           "tol": (b - a) * np.random.random_sample() + a,
                           "reg_covar": (d - c) * np.random.random_sample() + c,
                           "weight_concentration_prior": (0.1 - 1e-10) * np.random.random_sample() + 1e-10,
                           "mean_precision_prior": (0.1 - 1e-10) * np.random.random_sample() + 1e-10,
                           "n_components": sp_randint(2, 15),
                           "max_iter": sp_randint(10, 1000)}


        def run(self):
            print('Run Random Search ' + self.thread_name)
            cl = None
            if (self.thread_name == Constants.kmeans_algo):
                cl = KMeans()
            elif (self.thread_name == Constants.affinity_algo):
                cl = AffinityPropagation()
            elif (self.thread_name == Constants.mean_shift_algo):
                cl = MeanShift()
            elif (self.thread_name == Constants.ward_algo):
                cl = AgglomerativeClustering()
            elif (self.thread_name == Constants.dbscan_algo):
                cl = DBSCAN()
            elif (self.thread_name == Constants.gm_algo):
                cl = GaussianMixture()
            elif (self.thread_name == Constants.bgm_algo):
                cl = BayesianGaussianMixture()

            n_iter_search = 20
            random_search = RandomizedSearchCV(cl, param_distributions=self.clu_cs, n_iter=n_iter_search,
                                               scoring=make_scorer(my_custom_loss_func, greater_is_better=True))
            random_search.fit(self.X)
            self.value = random_search.best_score_
            self.parameters = random_search.best_params_

        def my_custom_loss_func(X, labels):
            labels_unique = np.unique(labels)
            n_clusters = len(labels_unique)
            value = Metric.metric(X, n_clusters, labels, self.metric)
            print("metric = " + str(value))
            return value
