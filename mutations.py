import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist


def one_nth_change_mutation(indiv: dict) -> dict:
    labels, data = indiv["labels"], indiv["data"]
    numbers_to_change = np.zeros(len(labels), dtype='bool')
    numbers_to_change[np.random.randint(len(labels))] = 1
    numbers_to_change[np.random.randint(len(labels), size=len(labels)) == 0] |= True
    cluster_sizes = np.bincount(labels)
    n_clusters = len(cluster_sizes)
    if np.random.randint(n_clusters ** 2) == 0:
        n_clusters += 1
    labels = labels.copy()
    labels[numbers_to_change] = np.random.randint(n_clusters, size=np.count_nonzero(numbers_to_change))
    empty = np.cumsum(np.bincount(labels) == 0)
    labels -= empty[labels]
    indiv = indiv.copy()
    indiv["labels"] = labels
    return indiv


def split_merge_move_mutation(indiv: dict) -> dict:
    labels, data = indiv["labels"], indiv["data"]
    cluster_sizes = np.bincount(labels)
    labels = labels.copy()
    while True:
        method = np.random.randint(3)
        if method == 0:
            ex_cluster = np.random.choice(np.argwhere(cluster_sizes > 1).flatten())
            centroid = data[labels == ex_cluster].mean(axis=0)
            norm = np.random.multivariate_normal(np.zeros(len(centroid)), np.identity(len(centroid)))
            dots = (data[labels == ex_cluster] - centroid).dot(norm)
            ratio = np.random.uniform(dots.min(), dots.max())
            negativeDot = dots < ratio
            negativeSum = np.count_nonzero(negativeDot)
            if negativeSum == 0 or negativeSum == len(negativeDot):
                continue
            labels[labels == ex_cluster] = np.where(negativeDot, len(cluster_sizes), ex_cluster)
        elif method == 1 and len(cluster_sizes) != 2:
            centroids = np.array([data[labels == i].mean(axis=0) for i in range(len(cluster_sizes))])
            dists = pdist(centroids, 'minkowski', p=1)
            dists = np.exp(-dists - np.log(np.exp(-dists).sum()))
            # dists = dists.max() - dists
            dists /= dists.sum()
            pair = np.random.choice(len(dists), p=dists)
            dists = np.zeros(len(dists))
            dists[pair] = 1
            src_cluster, dst_cluster = np.argwhere(squareform(dists) == 1)[0]
            # src_cluster, dst_cluster = np.random.choice(len(cluster_sizes), 2, replace=False)
            labels[labels == src_cluster] = dst_cluster
            labels[labels > src_cluster] -= 1
        else:
            # src_cluster = np.random.randint(len(cluster_sizes))
            dst_cluster = np.random.randint(len(cluster_sizes))  # - 1)
            # if dst_cluster >= src_cluster:
            #    dst_cluster += 1
            centroid = data[labels == dst_cluster].mean(axis=0)
            otherElems = labels != dst_cluster  # == src_cluster
            dists = np.linalg.norm(data[otherElems] - centroid, ord=1, axis=1)
            # dists = dists.max() - dists
            dists = np.exp(-dists - np.log(np.exp(-dists).sum()))
            dists /= dists.sum()
            n_points = np.count_nonzero(dists)
            n_points = np.random.binomial(n_points - 1, 1 / (n_points - 1)) + 1
            if n_points == np.count_nonzero(otherElems):
                continue
            n_indices = np.random.choice(np.argwhere(otherElems).flatten(), n_points, replace=False, p=dists)
            labels[n_indices] = dst_cluster
            emptyLabels = np.cumsum(np.bincount(labels) == 0)
            labels -= emptyLabels[labels]
        break
    indiv = indiv.copy()
    indiv["labels"] = labels
    return indiv


def split_eliminate_mutation(indiv: dict) -> dict:
    labels, data = indiv["labels"], indiv["data"]
    cluster_sizes = np.bincount(labels)
    labels = labels.copy()
    while True:
        method = np.random.randint(3)
        if method == 0:
            ex_cluster = np.random.choice(np.argwhere(cluster_sizes > 1).flatten())
            indices = np.argwhere(labels == ex_cluster).flatten()
            centroid = data[indices].mean(axis=0)
            centroid_dists = np.linalg.norm(data[indices] - centroid, axis=1)
            farthest = data[indices[np.argmax(centroid_dists)]]
            farthest_dists = np.linalg.norm(data[indices] - farthest, axis=1)
            n_cluster = farthest_dists < centroid_dists
            n_cluster_cnt = np.count_nonzero(n_cluster)
            if n_cluster_cnt == 0 or n_cluster_cnt == len(cluster_sizes):
                continue
            labels[labels == ex_cluster] = np.where(n_cluster, len(cluster_sizes), ex_cluster)
        elif method == 1 and len(cluster_sizes) > 2:
            centroids = np.array([data[labels == i].mean(axis=0) for i in range(len(cluster_sizes))])
            ex_cluster = np.random.randint(len(cluster_sizes))
            indices = np.argwhere(labels == ex_cluster).flatten()
            centroids = centroids[np.arange(0, len(cluster_sizes)) != ex_cluster]
            neighbours = np.argmin(cdist(data[indices], centroids), axis=1)
            neighbours[neighbours >= ex_cluster] += 1
            labels[indices] = neighbours
            labels[labels > ex_cluster] -= 1
        else:
            dst_cluster = np.random.randint(len(cluster_sizes))
            centroid = data[labels == dst_cluster].mean(axis=0)
            otherElems = labels != dst_cluster
            dists = np.linalg.norm(data[otherElems] - centroid, ord=1, axis=1)
            dists = np.exp(-dists - np.log(np.exp(-dists).sum()))
            dists /= dists.sum()
            n_points = np.count_nonzero(dists)
            n_points = np.random.binomial(n_points - 1, 1 / (n_points - 1)) + 1 if n_points > 1 else 1
            if n_points == np.count_nonzero(otherElems):
                continue
            n_indices = np.random.choice(np.argwhere(otherElems).flatten(), n_points, replace=False, p=dists)
            labels[n_indices] = dst_cluster
            emptyLabels = np.cumsum(np.bincount(labels) == 0)
            labels -= emptyLabels[labels]
        break
    indiv = indiv.copy()
    indiv["labels"] = labels
    return indiv
