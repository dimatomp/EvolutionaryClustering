import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
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


def density_based_dists_and_internals(dists, labels, d=2):
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
        if np.isnan(coredist).any():
            assert False
    reach_dists = squareform(np.maximum(coredist[:, None], coredist[None, :]), checks=False)
    reach_dists = np.maximum(reach_dists, dists)
    cluster_internals = np.zeros(len(labels), dtype='bool')
    internal_nodes_list = []
    msts = []
    for l, m in enumerate(cluster_masks):
        n_edges = np.count_nonzero(m)
        mst = minimum_spanning_tree(squareform(reach_dists[m])).toarray()
        mst = np.maximum(mst, mst.T)
        internal_nodes = np.count_nonzero(mst, axis=1) != 1 if n_edges != 1 else np.ones(mst.shape[0], dtype='bool')
        cluster_internals[labels == l] = internal_nodes
        internal_nodes_list.append(squareform(internal_nodes[None, :] | internal_nodes[:, None], checks=False))
        msts.append(squareform(mst))
    return reach_dists, cluster_internals, internal_nodes_list, msts


def density_based_internal_dists(dists, labels, reach_dists=None, cluster_internals=None, d=2):
    if reach_dists is None or cluster_internals is None:
        reach_dists, cluster_internals = density_based_dists_and_internals(dists, labels, d=d)[:2]
    internal_dists = reach_dists[squareform(cluster_internals[:, None] & cluster_internals[None, :], checks=False)]
    internal_labels = labels[cluster_internals]
    n_clusters = len(np.unique(labels))
    internal_matrices = internal_labels[None, :] == np.arange(0, n_clusters)[:, None]
    return internal_labels, internal_dists, internal_matrices


def density_based_separation_cohesion(labels, indiv, ord=2, **kwargs):
    internal_labels, internal_dists, internal_matrices = density_based_internal_dists(cache_distances(indiv, ord=ord),
                                                                                      labels)
    matrices = internal_matrices[:, None, :, None] & internal_matrices[None, :, None, :]
    matrices = matrices[squareform_matrix(matrices.shape[0]), :, :]
    np.logical_or(matrices, np.swapaxes(matrices, 1, 2), out=matrices)
    matrices = matrices[:, squareform_matrix(matrices.shape[2])]
    internal_dists = internal_dists * matrices
    return np.where(internal_dists != 0, internal_dists, np.inf).min(axis=1)


def density_based_cluster_sparseness(dists, labels, internal_nodes_list=None, msts=None, d=2):
    if internal_nodes_list is None or msts is None:
        internal_nodes_list, msts = density_based_dists_and_internals(dists, labels, d=d)[2:]
    return np.array([mst[internal_nodes].max() if internal_nodes.any() else 0 for mst, internal_nodes in
                     zip(msts, internal_nodes_list)])


def density_based_sparseness_separation(labels, indiv, ord=2, **kwargs):
    return density_based_cluster_sparseness(cache_distances(indiv, ord=ord), labels)


def density_based_cluster_validity(dists, labels, d=2):
    # ignore_points = cluster_counts[labels] > 1
    # if not ignore_points.all():
    #     ignore_clusters = np.cumsum(cluster_counts == 1)
    #     print('Encountered', ignore_clusters[-1], 'single-element clusters', file=sys.stderr)
    #     labels = labels[ignore_points]
    #     labels -= ignore_clusters[labels]
    #     dists = dists[squareform(ignore_points[:, None] & ignore_points[None, :], checks=False)]
    #     cluster_counts = cluster_counts[cluster_counts > 1]
    reach_dists, cluster_internals, internal_nodes_list, msts = density_based_dists_and_internals(dists, labels, d=d)
    cluster_sparseness = density_based_cluster_sparseness(dists, labels, internal_nodes_list, msts, d=d)
    internal_labels, internal_dists, internal_matrices = density_based_internal_dists(dists, labels,
                                                                                      reach_dists=reach_dists,
                                                                                      cluster_internals=cluster_internals,
                                                                                      d=d)
    matrices = np.logical_xor(internal_matrices[:, :, None], internal_matrices[:, None, :])
    matrices = matrices[:, squareform_matrix(len(internal_labels))]
    internal_dists = internal_dists * matrices
    cluster_separation = np.where(internal_dists != 0, internal_dists, np.inf).min(axis=1)
    return (cluster_separation - cluster_sparseness) / np.maximum(cluster_separation, cluster_sparseness)


def density_based_validity_separation(labels, indiv, ord=2, **kwargs):
    result = density_based_cluster_validity(cache_distances(indiv, ord=ord), labels, d=2)
    return result.max() - result


def centroid_distance_cohesion(ord=2, **kwargs):
    clusters, centroids = get_clusters_and_centroids(**kwargs)
    return pdist(centroids, metric='minkowski', p=ord)


all_cohesions = ['centroid_distance_cohesion', 'density_based_separation_cohesion']
all_separations = ['density_based_validity_separation', 'density_based_sparseness_separation',
                   'mean_centroid_distance_separation', 'diameter_separation']


def construct_probabilities(values, the_less_the_better=True):
    # values = np.exp(-values - np.log(np.exp(-values).sum()))
    if len(values) == 1:
        return np.ones(1)
    if the_less_the_better:
        values = values.max() - values
    return values / values.sum()
