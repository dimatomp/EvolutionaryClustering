from sklearn import datasets
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace import InCondition

# Import SMAC-utilities

from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

from . import Metric
from . import Constants

#train_datas = pd.read_csv('aaa.csv', header=0)
#iris = datasets.load_iris()#
#X_all = iris.data
#y_all = iris.target

n_samples = 500

# 2 nonoverlap circlues
#noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
#X_all, y_all = noisy_circles

# 2 nonoverlap
#noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
#X_all, y_all = noisy_moons

# 3 strange
#X_all, y_all = datasets.make_blobs(n_samples=n_samples, random_state=170)
#transformation = [[0.4, -0.5], [-0.4, 0.8]]
#X = np.dot(X_all, transformation)

#varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[0.1, 2.5, 0.5])
#X_all, y_all = varied

# 3 easily observed clusters
X, y = datasets.make_blobs(n_samples=n_samples, n_features=4, centers=3, cluster_std=1, center_box=(-10.0, 10.0), shuffle=True)

#X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25)
#X = X_all

max_eval = 10
metric = Constants.dunn_metric

def run(cl):
    cl.fit(X)
    labels = cl.labels_
    #centers = cl.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    print("n_clusters = " + str(n_clusters))
    m = Metric.metric(X, n_clusters, labels, metric)
    print("metric = " + str(m))
    return m

def km_run(cfg):
    cl = KMeans(**cfg)
    return run(cl)

def ms_run(cfg):
    bandwidth = estimate_bandwidth(X, quantile=cfg['quantile'])
    cl = MeanShift(bandwidth=bandwidth, bin_seeding=bool(cfg['bin_seeding']), min_bin_freq=cfg['min_bin_freq'], cluster_all=bool(cfg['cluster_all']))
    return run(cl)

def aff_run(cfg):
    cl = AffinityPropagation(**cfg)
    return run(cl)

def w_run(cfg):
    cl = AgglomerativeClustering(**cfg)
    return run(cl)

def db_run(cfg):
    cl = DBSCAN(**cfg)
    return run(cl)

# Build Configuration Space which defines all parameters and their ranges
km_cs = ConfigurationSpace()
algorithm = CategoricalHyperparameter("algorithm", ["auto", "full", "elkan"], default="auto")
tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2, default=1e-4)
n_clusters = UniformIntegerHyperparameter("n_clusters", 2, 15, default=8)
n_init = UniformIntegerHyperparameter("n_init", 2, 15, default=10)
max_iter = UniformIntegerHyperparameter("max_iter", 50, 1500, default=300)
verbose = UniformIntegerHyperparameter("verbose", 0, 10, default=0)
km_cs.add_hyperparameters([n_clusters, n_init, max_iter, tol, verbose, algorithm])

km_scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": max_eval,  # maximum function evaluations
                     "cs": km_cs,               # configuration space
                     "deterministic": "true"
                     })

ms_cs = ConfigurationSpace()
quantile = UniformFloatHyperparameter("quantile", 0.0, 1.0, default=0.3)
bin_seeding = UniformIntegerHyperparameter("bin_seeding", 0, 1, default=0)
min_bin_freq = UniformIntegerHyperparameter("min_bin_freq", 1, 100, default=1)
cluster_all = UniformIntegerHyperparameter("cluster_all", 0, 1, default=0)
ms_cs.add_hyperparameters([quantile, bin_seeding, min_bin_freq, cluster_all])

ms_scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": max_eval,  # maximum function evaluations
                     "cs": ms_cs,               # configuration space
                     "deterministic": "true"
                     })

aff_cs = ConfigurationSpace()
damping = UniformFloatHyperparameter("damping", 0.5, 1.0, default=0.5)
max_iter = UniformIntegerHyperparameter("max_iter", 100, 1000, default=200)
convergence_iter = UniformIntegerHyperparameter("convergence_iter", 5, 20, default=15)
aff_cs.add_hyperparameters([damping, max_iter, convergence_iter])

aff_scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": max_eval,  # maximum function evaluations
                     "cs": aff_cs,               # configuration space
                     "deterministic": "true"
                     })

w_cs = ConfigurationSpace()
affinity = CategoricalHyperparameter("affinity", ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"], default="euclidean")
linkage = CategoricalHyperparameter("linkage", ["ward", "complete", "average"], default="ward")
n_clusters = UniformIntegerHyperparameter("n_clusters", 2, 15, default=2)
w_cs.add_hyperparameters([n_clusters, affinity, linkage])

w_scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": max_eval,  # maximum function evaluations
                     "cs": w_cs,               # configuration space
                     "deterministic": "true"
                     })

db_cs = ConfigurationSpace()
algorithm = CategoricalHyperparameter("algorithm", ["auto", "ball_tree", "kd_tree", "brute"], default="auto")
eps = UniformFloatHyperparameter("eps", 0.1, 0.9, default=0.5)
min_samples = UniformIntegerHyperparameter("min_samples", 2, 10, default=5)
leaf_size = UniformIntegerHyperparameter("leaf_size", 5, 100, default=30)
db_cs.add_hyperparameters([eps, min_samples, algorithm, leaf_size])

db_scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": max_eval,  # maximum function evaluations
                     "cs": db_cs,               # configuration space
                     "deterministic": "true"
                     })

best_val = 1.0
best_algo = "-1"
best_params = dict()

i = 0
algos = {Constants.dbscan_algo:0, Constants.kmeans_algo:0, Constants.affinity_algo:0, Constants.mean_shift_algo:0, Constants.ward_algo:0}
metrics = [Constants.davies_bouldin_metric, Constants.dunn_metric, Constants.cal_har_metric, Constants.silhouette_metric,
           Constants.dunn31_metric, Constants.dunn41_metric, Constants.dunn51_metric, Constants.dunn33_metric, Constants.dunn43_metric,
           Constants.dunn53_metric]
# metrics = ["sc"]
# algos = {Constants.kmeans_algo:0}
saved_parameters = [""] * len(metrics)
num_parameters_for_algo = {Constants.kmeans_algo:[], Constants.affinity_algo:[], Constants.mean_shift_algo:[], Constants.ward_algo:[], Constants.dbscan_algo:[]}

for metric in metrics:
    for algo in algos.keys():
        value = 1
        parameters = ""
        if (Constants.kmeans_algo in algo):
            smac = SMAC(scenario=km_scenario, rng=np.random.RandomState(42), tae_runner=km_run)
            parameters = smac.optimize()
            value = km_run(parameters)
        elif (Constants.affinity_algo in algo):
            smac = SMAC(scenario=aff_scenario, rng=np.random.RandomState(42), tae_runner=aff_run)
            parameters = smac.optimize()
            value = aff_run(parameters)
        elif (Constants.mean_shift_algo in algo):
            smac = SMAC(scenario=ms_scenario, rng=np.random.RandomState(42), tae_runner=ms_run)
            parameters = smac.optimize()
            value = ms_run(parameters)
        elif (Constants.ward_algo in algo):
            smac = SMAC(scenario=w_scenario, rng=np.random.RandomState(42), tae_runner=w_run)
            parameters = smac.optimize()
            value = w_run(parameters)
        elif (Constants.dbscan_algo in algo):
            smac = SMAC(scenario=db_scenario, rng=np.random.RandomState(42), tae_runner=db_run)
            parameters = smac.optimize()
            value = db_run(parameters)
        print(('For algo ' + algo + ' lowest function value found: %f' % value))
        print(('Parameter setting %s' % parameters))
        if (value < best_val):
            best_val = value
            best_algo = algo
            best_params = parameters
    algos[best_algo] += 1
    saved_parameters[i] = best_params
    num_parameters_for_algo[best_algo].append(i)
    i += 1

chosen_algo = ""
num_cases = 0
for algo in algos.keys():
    if (algos[algo] > num_cases):
        num_cases = algos[algo]
        chosen_algo = algo

best_params = saved_parameters[num_parameters_for_algo[chosen_algo][0]]
cl = ""
if   (Constants.kmeans_algo in chosen_algo):
    cl = KMeans(**best_params)
elif (Constants.affinity_algo in chosen_algo):
    cl = AffinityPropagation(**best_params)
elif (Constants.mean_shift_algo in chosen_algo):
    bandwidth = estimate_bandwidth(X, quantile=best_params["quantile"])
    cl = MeanShift(bandwidth=bandwidth, bin_seeding=bool(best_params["bin_seeding"]), min_bin_freq=best_params["min_bin_freq"],
              cluster_all=bool(best_params["cluster_all"]))
elif (Constants.ward_algo in chosen_algo):
    cl = AgglomerativeClustering(**best_params)
elif (Constants.dbscan_algo in chosen_algo):
    cl = DBSCAN(**best_params)

cl.fit(X)

# Uncomment to get graphics
# plt.figure(figsize=(15, 10))
# plt.subplot(221)
# plt.scatter(X[:, 0], X[:, 1], c=cl.labels_)
# plt.title("Predicted lables")

#plt.subplot(222)
#plt.scatter(X[:, 0], X[:, 3], c=y_all)
#plt.title("True lables")

print('Best algorithm = ' + chosen_algo)
#print('Best metric = ' + str(best_val))
print('Best parameters = ' + str(best_params))
print(str(algos))
#print(algo.score(X_test, y_test))


# Uncomment to show:
# plt.show()