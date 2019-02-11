import numpy as np
from numpy.linalg import norm


def get_clusters_and_centroids(labels=None, data=None, cluster_labels=None, clusters=None, centroids=None, **kwargs):
    if clusters is None or centroids is None:
        if cluster_labels is None:
            cluster_labels = np.unique(labels)
        clusters = [data[labels == i] for i in cluster_labels]
        centroids = np.array([d.mean(axis=0) for d in clusters])
    return clusters, centroids


def diameter_separation(**kwargs):
    clusters, centroids = get_clusters_and_centroids(**kwargs)
    points = [d[np.argmax(norm(d - c, axis=1))] for d, c in zip(clusters, centroids)]
    return [norm(d - p, axis=1).max() for d, p in zip(clusters, points)]


def mean_centroid_distance_separation(**kwargs):
    clusters, centroids = get_clusters_and_centroids(**kwargs)
    return [norm(cluster - centroid).mean() for cluster, centroid in zip(clusters, centroids)]
