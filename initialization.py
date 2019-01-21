import numpy as np


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
