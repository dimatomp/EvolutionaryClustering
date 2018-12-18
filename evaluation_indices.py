import numpy as np
from scipy.spatial.distance import pdist, squareform


def silhouette_index(indiv):
    values, data = indiv
    sortingPerm = np.argsort(values)
    clusterCounts = np.bincount(values)
    sortedValues = values[sortingPerm][None, :]
    dists = squareform(pdist(data[sortingPerm])) / clusterCounts[sortedValues[0]][None, :]
    a = np.where(sortedValues == sortedValues.T, dists, 0).sum(axis=1)
    dists = np.where(sortedValues != sortedValues.T, dists, np.inf)
    b = np.vstack([np.where(sortedValues == i, dists, 0).sum(axis=1) for i in range(len(clusterCounts))]).min(axis=0)
    return ((b - a) / np.maximum(a, b)).mean()
