import pandas as pd

import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import matplotlib.pyplot as plt

from .. import Constants
from .. import Metric

metrics = ["calinski-harabasz", "silhouette", "cop"]
ds_by_size = [
    "iris",
    "glass",
    # "haberman",
    "wholesale",
    "indiandiabests",
    "yeast",
    "krvskp"
]


def evaluate(x, labels, metric):
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    m = Metric.metric(x, n_clusters, labels, metric)
    return m


def draw(x, labels, dataset, algo):
    X_res = PCA(n_components=2).fit_transform(x)
    plt.figure(figsize=(9, 6))

    plt.scatter(X_res[:, 0], X_res[:, 1], c=labels)
    plt.title(algo + "(" + dataset + "). %s clusters" % len(np.unique(labels)))

    plt.savefig('./def_pics/fig_' + dataset + "_" + algo)
    plt.clf()
    plt.close()


f = open(file="./def_pics/default.txt", mode='w')
f_std = open(file="./def_pics/default.stddev.txt", mode='w')

header = str("KMeans" + "\t"
             + "Affinity_Propagation" + "\t"
             + "Mean_Shift" + "\t"
             + "Ward" + "\t"
             + "DBSCAN" + "\t"
             + "Gaussian_Mixture" + "\t"
             + "Bayesian_Gaussian_Mixture" + "\n")

f.write(header)
f_std.write(header)

for dataset in ds_by_size:
    print("\t" + dataset + "...")
    data = pd.read_csv(Constants.experiment_path + dataset + ".csv")
    # data = pd.read_csv("datasets/unified/" + dataset + ".csv")
    X = np.array(data, dtype=np.double)

    cl = KMeans()
    labels = cl.fit_predict(X)
    draw(X, labels, dataset, "KMeans")

    cl = AffinityPropagation()
    labels = cl.fit_predict(X)
    draw(X, labels, dataset, "Aff. Prop.")

    cl = MeanShift()
    labels = cl.fit_predict(X)
    draw(X, labels, dataset, "Mean Shift")

    cl = AgglomerativeClustering()
    labels = cl.fit_predict(X)
    draw(X, labels, dataset, "Agglomerative")

    cl = DBSCAN()
    labels = cl.fit_predict(X)
    draw(X, labels, dataset, "DBSCAN")

    cl = GaussianMixture()
    cl.fit(X)
    labels = cl.predict(X)
    draw(X, labels, dataset, "Gaussian")

    cl = BayesianGaussianMixture()
    cl.fit(X)
    labels = cl.predict(X)
    draw(X, labels, dataset, "Bayessian Gaussian")

for m in metrics:
    print(m + "...")
    for dataset in ds_by_size:
        print("\t" + dataset + "...")
        data = pd.read_csv(Constants.experiment_path + dataset + ".csv")
        X = np.array(data, dtype=np.double)

        results = [[] for i in range(0, Constants.num_algos)]
        for i in range(0, 10):
            cl = KMeans()
            labels = cl.fit_predict(X)
            results.append(evaluate(X, labels, m))

            cl = AffinityPropagation()
            labels = cl.fit_predict(X)
            results.append(evaluate(X, labels, m))

            cl = MeanShift()
            labels = cl.fit_predict(X)
            results.append(evaluate(X, labels, m))

            cl = AgglomerativeClustering()
            labels = cl.fit_predict(X)
            results.append(evaluate(X, labels, m))

            cl = DBSCAN()
            labels = cl.fit_predict(X)
            results.append(evaluate(X, labels, m))

            cl = GaussianMixture()
            cl.fit(X)
            labels = cl.predict(X)
            results.append(evaluate(X, labels, m))

            cl = BayesianGaussianMixture()
            cl.fit(X)
            labels = cl.predict(X)
            results.append(evaluate(X, labels, m))

        for i in range(0, Constants.num_algos):
            f.write(str(np.mean(results[i])) + "\t")
            f_std.write(str(np.std(results[i])) + "\t")

        f.write("\n")
