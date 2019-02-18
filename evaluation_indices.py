from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import silhouette_score, calinski_harabaz_score, davies_bouldin_score
from cluster_measures import *


def evaluation_index(minimize):
    def construct(idx):
        class EvaluationIndex:
            is_minimized = minimize

            def __call__(self, *args, **kwargs):
                return idx(*args, **kwargs)

        return EvaluationIndex()

    return construct


@evaluation_index(minimize=False)
def silhouette_index(indiv):
    return silhouette_score(indiv["data"], indiv["labels"])


@evaluation_index(minimize=False)
def calinski_harabaz_index(indiv):
    return calinski_harabaz_score(indiv["data"], indiv["labels"])


@evaluation_index(minimize=True)
def davies_bouldin_index(indiv):
    return davies_bouldin_score(indiv["data"], indiv["labels"])


def cache_distances(indiv):
    if "distances" in indiv:
        distances = indiv["distances"]
    else:
        distances = pdist(indiv["data"])
        indiv["distances"] = distances
    return distances


def squareform_matrix(n):
    squareform_m = np.arange(0, n)
    return squareform_m[None, :] > squareform_m[:, None]


def dvcb_index(d=2):
    @evaluation_index(minimize=False)
    def index(indiv):
        labels, data = indiv["labels"], indiv["data"]
        dists = cache_distances(indiv)
        cluster_counts = np.bincount(labels)
        # ignore_points = cluster_counts[labels] > 1
        # if not ignore_points.all():
        #     ignore_clusters = np.cumsum(cluster_counts == 1)
        #     print('Encountered', ignore_clusters[-1], 'single-element clusters', file=sys.stderr)
        #     labels = labels[ignore_points]
        #     labels -= ignore_clusters[labels]
        #     dists = dists[squareform(ignore_points[:, None] & ignore_points[None, :], checks=False)]
        #     cluster_counts = cluster_counts[cluster_counts > 1]
        coredist = np.zeros(len(labels))
        cluster_masks = []
        for l in range(len(cluster_counts)):
            labels_mask = labels == l
            cluster_mask = labels_mask[:, None] & labels_mask[None, :]
            cluster_mask = squareform(cluster_mask, checks=False)
            cluster_masks.append(cluster_mask)
            cluster_dist = squareform(dists[cluster_mask])
            cluster_dist = np.where(cluster_dist != 0, cluster_dist ** (-d), 0)
            coredist[labels_mask] = (cluster_dist.sum(axis=1) / (cluster_dist.shape[1] - 1)) ** (-1 / d)
        reach_dists = squareform(np.maximum(coredist[:, None], coredist[None, :]), checks=False)
        reach_dists = np.maximum(reach_dists, dists)
        cluster_sparseness = []
        cluster_internals = np.zeros(len(labels), dtype='bool')
        for l, m in enumerate(cluster_masks):
            n_edges = np.count_nonzero(m)
            mst = minimum_spanning_tree(squareform(reach_dists[m])).toarray()
            mst = np.maximum(mst, mst.T)
            internal_nodes = np.count_nonzero(mst, axis=1) != 1 if n_edges != 1 else np.ones(mst.shape[0], dtype='bool')
            cluster_internals[labels == l] = internal_nodes
            mst = squareform(mst)
            internal_nodes = squareform(internal_nodes[None, :] | internal_nodes[:, None], checks=False)
            cluster_sparseness.append(mst[internal_nodes].max() if internal_nodes.any() else 0)
        cluster_sparseness = np.array(cluster_sparseness)
        internal_dists = reach_dists[squareform(cluster_internals[:, None] & cluster_internals[None, :], checks=False)]
        internal_labels = labels[cluster_internals]
        matrices = internal_labels[None, :] == np.arange(0, len(cluster_counts))[:, None]
        matrices = np.logical_xor(matrices[:, :, None], matrices[:, None, :])
        matrices = matrices[:, squareform_matrix(len(internal_labels))]
        internal_dists = internal_dists * matrices
        cluster_separation = np.where(internal_dists != 0, internal_dists, np.inf).min(axis=1)
        cluster_validities = (cluster_separation - cluster_sparseness) / np.maximum(cluster_separation,
                                                                                    cluster_sparseness)
        return (cluster_counts * cluster_validities).sum() / len(labels)

    return index


def generalized_dunn_index(separation, cohesion):
    @evaluation_index(minimize=False)
    def index(indiv):
        labels, data = indiv["labels"], indiv["data"]
        cluster_labels = np.unique(labels)
        clusters = [data[labels == i] for i in cluster_labels]
        centroids = np.array([d.mean(axis=0) for d in clusters])

        if separation == "single_linkage":
            dists = cache_distances(indiv)
            min_distance = dists[squareform(labels[:, None] != labels[None, :])].min()
        elif separation == "mean_per_point":
            # TODO Too slow
            dists = cache_distances(indiv)
            if "squareform" in indiv:
                squareform_m = indiv["squareform"]
            else:
                squareform_m = squareform_matrix(len(labels))
                indiv["squareform"] = squareform_m
            matrices = labels[None, :] == cluster_labels[:, None]
            matrices = matrices[:, None, :, None] & matrices[None, :, None, :]
            square_cls = np.arange(0, len(cluster_labels))
            square_cls = square_cls[None, :] > square_cls[:, None]
            matrices = matrices[square_cls, :, :]
            np.logical_or(matrices, np.swapaxes(matrices, 1, 2), out=matrices)
            matrices = matrices[:, squareform_m]
            min_distance = ((dists * matrices).sum(axis=1) / np.count_nonzero(matrices, axis=1)).min()
        elif separation == "centroid_distance":
            min_distance = centroid_distance_cohesion(clusters=clusters, centroids=centroids).min()
        elif separation == "mean_per_cluster":
            min_distance = min(min(
                norm(cluster1 - centroid2).sum() + norm(cluster2 - centroid1).sum() / (len(cluster1) + len(cluster2))
                for
                cluster2, centroid2 in zip(clusters[idx + 1:], centroids[idx + 1:])) for idx, (cluster1, centroid1) in
                               enumerate(zip(clusters[:-1], centroids[:-1])))
        if cohesion == "diameter":
            max_diameter = max(diameter_separation(clusters=clusters, centroids=centroids))
        elif cohesion == "mean_distance":
            max_diameter = 2 * max(mean_centroid_distance_separation(clusters=clusters, centroids=centroids))
        return min_distance / max_diameter

    return index


dunn_index = generalized_dunn_index(separation="single_linkage", cohesion="diameter")

# TODO 3 more (density-based?) measures, not from [2009]
'''
class DynamicIndex:
    def __init__(self, index):
        self.index = index
        self.prev_labels = None

    def __call__(self, indiv):
        labels, data = indiv['labels'], indiv['data']
        n_clusters = len(np.unique(labels))
        if self.prev_labels is None:
            self.result = self.index.find(data, labels, n_clusters)
        else:
            for idx in np.argwhere(self.prev_labels != labels).flatten():
                self.result = self.index.update(data, n_clusters, labels, self.prev_labels[idx], labels[idx], idx)
        self.prev_labels = labels
        return self.result


class DynamicGeneralizedDunn31Index(DynamicIndex):
    is_minimized = False

    def __init__(self):
        super(DynamicGeneralizedDunn31Index, self).__init__(gD31_index.Index())
'''
