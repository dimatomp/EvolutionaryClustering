import time

import numpy as np
import pandas as pd
from ConfigSpace import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from sklearn.cluster import KMeans
from smac.configspace import ConfigurationSpace

from .. import Constants
from .. import Metric

config_space = ConfigurationSpace()
algorithm = CategoricalHyperparameter("algorithm", ["auto", "full", "elkan"])
tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2)
n_clusters = UniformIntegerHyperparameter("n_clusters", 2, 15)
n_init = UniformIntegerHyperparameter("n_init", 2, 15)
max_iter = UniformIntegerHyperparameter("max_iter", 50, 1500)
config_space.add_hyperparameters([n_clusters, n_init, max_iter, tol, algorithm])
cfg = config_space.sample_configuration().get_dictionary()

metric = "s-dbw"

data = pd.read_csv( Constants.experiment_path + "iris.csv")
X = np.array(data, dtype=np.double)

cl = KMeans(**cfg)
labels = cl.fit_predict(X)
labels_unique = np.unique(labels)
n_clusters = len(labels_unique)

start = time.time()
m = Metric.metric(X, n_clusters, labels, metric)

print("Time spent: " + str(time.time() - start))

print(m)
