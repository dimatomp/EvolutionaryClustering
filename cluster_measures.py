import numpy as np
import sys
from numpy.linalg import norm
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.sparse import find, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
from individual import Individual


def get_clusters_and_centroids(labels=None, data=None, cluster_labels=None, clusters=None, centroids=None, **kwargs):
    if clusters is None or centroids is None:
        if cluster_labels is None:
            cluster_labels = np.unique(labels)
        clusters = [data[labels == i] for i in cluster_labels]
        centroids = np.array([d.mean(axis=0) for d in clusters])
    return clusters, centroids


def cleanup_empty_clusters(labels):
    labels -= np.cumsum(np.bincount(labels) == 0)[labels]


def cache_distances(indiv: Individual, ord=2):
    name = "distances" if ord == 2 else "distances_" + str(ord)
    if name in indiv:
        distances = indiv[name]
    else:
        distances = pdist(indiv["data"])
        indiv.set_data_field(name, distances)
    return distances


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
    return np.array([norm(d - p, axis=1).max() for d, p in zip(clusters, points)])


def mean_centroid_distance_separation(ord=2, **kwargs):
    clusters, centroids = get_clusters_and_centroids(**kwargs)
    return np.array([norm(cluster - centroid, ord=ord).mean() for cluster, centroid in zip(clusters, centroids)])


def squareform_matrix(n):
    squareform_m = np.arange(0, n)
    return squareform_m[None, :] > squareform_m[:, None]


def density_based_dists_and_internals(dists, labels, d):
    n_clusters = len(np.unique(labels))
    coredist = np.zeros(len(labels))
    cluster_masks = []
    for l in range(n_clusters):
        labels_mask = labels == l
        cluster_mask = labels_mask[:, None] & labels_mask[None, :]
        cluster_mask = squareform(cluster_mask, checks=False)
        cluster_masks.append(cluster_mask)
        cluster_dist = squareform(dists[cluster_mask])
        cluster_dist = np.where(cluster_dist != 0, cluster_dist ** (-d), 0)
        if (np.count_nonzero(labels_mask)) > 1:
            coredist[labels_mask] = (cluster_dist.sum(axis=1) / (cluster_dist.shape[1] - 1)) ** (-1 / d)
        assert not np.isnan(coredist).any() and (coredist != np.inf).all()
    reach_dists = squareform(np.maximum(coredist[:, None], coredist[None, :]), checks=False)
    reach_dists = np.maximum(reach_dists, dists)
    mst = minimum_spanning_tree(squareform(reach_dists))
    rows, cols = mst.nonzero()
    rows, cols, vals = find(mst)
    mst = csr_matrix((np.hstack([vals, vals]), (np.hstack([rows, cols]), np.hstack([cols, rows]))), shape=(len(labels), len(labels)))
    rows, cols = np.bincount(rows), np.bincount(cols)
    if len(rows) != len(labels):
       rows = np.hstack([rows, np.zeros(len(labels) - len(rows), dtype='int')])
    if len(cols) != len(labels):
       cols = np.hstack([cols, np.zeros(len(labels) - len(cols), dtype='int')])
    cluster_internals = rows + cols != 1
    return cluster_internals, mst, n_clusters
    # return mst, n_clusters


def density_based_internal_dists(labels, cluster_internals, n_clusters, mst):
    internal_labels = labels[cluster_internals]
# def density_based_internal_dists(labels, n_clusters, mst):
#     internal_labels = labels
    internal_matrices = internal_labels[None, :] == np.arange(0, n_clusters)[:, None]
    paths = dijkstra(mst, directed=False)
    paths = paths[cluster_internals][:, cluster_internals]
    return internal_labels, internal_matrices, squareform(paths, checks=False)


def density_based_separation_cohesion(labels, indiv, ord=2, **kwargs):
    cluster_internals, mst, n_clusters = density_based_dists_and_internals(cache_distances(indiv, ord=ord), labels,
                                                                           indiv['data'].shape[1])
    # mst, n_clusters = density_based_dists_and_internals(cache_distances(indiv, ord=ord), labels, indiv['data'].shape[1])
    internal_labels, internal_matrices, paths = density_based_internal_dists(labels, cluster_internals, n_clusters, mst)
    # internal_labels, internal_matrices, paths = density_based_internal_dists(labels, n_clusters, mst)
    #try:
    matrices = internal_matrices[:, None, :, None] & internal_matrices[None, :, None, :]
    matrices = matrices[squareform_matrix(matrices.shape[0]), :, :]
    np.logical_or(matrices, np.swapaxes(matrices, 1, 2), out=matrices)
    matrices = matrices[:, squareform_matrix(matrices.shape[2])]
    internal_dists = np.where(matrices, paths, np.inf)
    result = internal_dists.min(axis=1)
    return np.where(matrices.any(axis=1), result, result.min())
    #except MemoryError:
    #    print('Measuring density-based cohesion slowly', file=sys.stderr)
    #    ans = []
    #    for cl1 in range(n_clusters):
    #        labels1 = internal_labels == cl1
    #        for cl2 in range(cl1 + 1, n_clusters):
    #            both_labels = labels1[:, None] & (internal_labels == cl2)[None, :]
    #            ans.append(paths[squareform(both_labels | both_labels.T)].min())
    #    return np.array(ans)


def density_based_cluster_sparseness(dists, labels, d, n_clusters=None, mst=None, cluster_internals=None):
    if n_clusters is None or mst is None or cluster_internals is None:
        cluster_internals, mst, n_clusters = density_based_dists_and_internals(dists, labels, d)
        # mst, n_clusters = density_based_dists_and_internals(dists, labels, d)
    internal_nodes_list = (np.arange(0, n_clusters)[:, None] == labels[None, :]) & cluster_internals
    return np.array([mst[internal_nodes][:, internal_nodes].max() if internal_nodes.any() else 0 for internal_nodes in
                     internal_nodes_list])
    # return np.array([mst[internal_nodes][:, internal_nodes].max() for internal_nodes in internal_nodes_list])


def density_based_sparseness_separation(labels, indiv, ord=2, **kwargs):
    return density_based_cluster_sparseness(cache_distances(indiv, ord=ord), labels, indiv['data'].shape[1])


def density_based_cluster_validity(dists, labels, d, return_intcount=False):
    # ignore_points = cluster_counts[labels] > 1
    # if not ignore_points.all():
    #     ignore_clusters = np.cumsum(cluster_counts == 1)
    #     print('Encountered', ignore_clusters[-1], 'single-element clusters', file=sys.stderr)
    #     labels = labels[ignore_points]
    #     labels -= ignore_clusters[labels]
    #     dists = dists[squareform(ignore_points[:, None] & ignore_points[None, :], checks=False)]
    #     cluster_counts = cluster_counts[cluster_counts > 1]
    cluster_internals, mst, n_clusters = density_based_dists_and_internals(dists, labels, d)
    # mst, n_clusters = density_based_dists_and_internals(dists, labels, d)
    cluster_sparseness = density_based_cluster_sparseness(dists, labels, d, n_clusters=n_clusters,
                                                          mst=mst, cluster_internals=cluster_internals)
    internal_labels, internal_matrices, paths = density_based_internal_dists(labels, cluster_internals, n_clusters, mst)
    # internal_labels, internal_matrices, paths = density_based_internal_dists(labels, n_clusters, mst)
    matrices = np.logical_xor(internal_matrices[:, :, None], internal_matrices[:, None, :])
    matrices = matrices[:, squareform_matrix(len(internal_labels))]
    internal_dists = np.where(matrices, paths, np.inf)
    cluster_separation = internal_dists.min(axis=1)
    result = np.where(matrices.any(axis=1), (cluster_separation - cluster_sparseness) / np.maximum(cluster_separation, cluster_sparseness), -1)
    if not return_intcount:
        return result
    intcount = np.bincount(internal_labels)
    if len(intcount) < len(result):
        intcount = np.hstack([intcount, np.zeros(len(result) - len(intcount))])
    return result, intcount


def density_based_validity_separation(labels, indiv, ord=2, **kwargs):
    result = density_based_cluster_validity(cache_distances(indiv, ord=ord), labels, indiv['data'].shape[1])
    return 1 - result


def centroid_distance_cohesion(ord=2, **kwargs):
    clusters, centroids = get_clusters_and_centroids(**kwargs)
    return pdist(centroids, metric='minkowski', p=ord)


all_cohesions = ['centroid_distance_cohesion', 'density_based_separation_cohesion']
all_separations = ['density_based_validity_separation', 'density_based_sparseness_separation',
                   'mean_centroid_distance_separation', 'diameter_separation']


def construct_probabilities(values, the_less_the_better=True):
    # values = np.exp(-values - np.log(np.exp(-values).sum()))
    if len(np.unique(values)) == 1:
        return np.ones(len(values)) / len(values)
    if the_less_the_better:
        values = values.max() + values.min() - values
    return values / values.sum()
