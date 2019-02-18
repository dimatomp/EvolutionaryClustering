import numpy as np
from cluster_measures import *


def random_initialization(data, n_clusters):
    return {"labels": np.random.randint(low=0, high=n_clusters, size=len(data)), "data": data}


def axis_initialization(data, n_clusters):
    centroid = data.mean(axis=0)
    norm = np.random.multivariate_normal(np.zeros(len(centroid)), np.identity(len(centroid)))
    dots = (data - centroid).dot(norm)
    dotmin = dots.min()
    labels = np.minimum(np.floor((dots - dotmin) / (dots.max() - dotmin) * n_clusters).astype('int'),
                        n_clusters - 1)
    emptyLabels = np.cumsum(np.bincount(labels) == 0)
    labels -= emptyLabels[labels]
    return {"labels": labels, "data": data}


def centroid_initialization(data, n_clusters):
    datamin, datamax = data.min(axis=0), data.max(axis=0)
    centroids = np.random.sample((n_clusters, data.shape[1])) * (datamax - datamin) + datamin
    labels, centroids = get_labels_by_centroids(centroids, data)
    return {"labels": labels, "data": data, "centroids": centroids}


def prototype_initialization(data, n_clusters):
    prototypes = np.zeros(len(data), dtype='bool')
    prototypes[np.random.choice(len(data), n_clusters)] = True
    return {"labels": get_labels_by_prototypes(prototypes, data), "prototypes": prototypes, "data": data}
