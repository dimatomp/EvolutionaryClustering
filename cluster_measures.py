import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import pdist, cdist


def get_clusters_and_centroids(labels=None, data=None, cluster_labels=None, clusters=None, centroids=None, **kwargs):
    if clusters is None or centroids is None:
        if cluster_labels is None:
            cluster_labels = np.unique(labels)
        clusters = [data[labels == i] for i in cluster_labels]
        centroids = np.array([d.mean(axis=0) for d in clusters])
    return clusters, centroids


def cleanup_empty_clusters(labels):
    labels -= np.cumsum(np.bincount(labels) == 0)[labels]


def get_labels_by_centroids(centroids, data):
    labels = cdist(data, centroids).argmin(axis=1)
    centroids = centroids[np.isin(np.arange(0, len(centroids)), labels)]
    cleanup_empty_clusters(labels)
    return labels, centroids


def get_labels_by_prototypes(prototypes, data):
    return cdist(data, data[prototypes]).argmin(axis=1)


def diameter_separation(ord=2, **kwargs):
    clusters, centroids = get_clusters_and_centroids(**kwargs)
    points = [d[np.argmax(norm(d - c, axis=1, ord=ord))] for d, c in zip(clusters, centroids)]
    return [norm(d - p, axis=1).max() for d, p in zip(clusters, points)]


def mean_centroid_distance_separation(ord=2, **kwargs):
    clusters, centroids = get_clusters_and_centroids(**kwargs)
    return np.array([norm(cluster - centroid, ord=ord).mean() for cluster, centroid in zip(clusters, centroids)])


def centroid_distance_cohesion(ord=2, **kwargs):
    clusters, centroids = get_clusters_and_centroids(**kwargs)
    return pdist(centroids, metric='minkowski', p=ord)


def construct_probabilities(values):
    # values = np.exp(-values - np.log(np.exp(-values).sum()))
    values = values.max() - values
    return values / values.sum()
