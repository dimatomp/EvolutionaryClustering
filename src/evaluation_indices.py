from sklearn.metrics import silhouette_score, calinski_harabaz_score
from .cluster_measures import *


def evaluation_index(minimize):
    def construct(idx):
        class EvaluationIndex:
            is_minimized = minimize

            def __str__(self):
                return idx.__name__

            def __call__(self, indiv, *args, **kwargs):
                if len(np.unique(indiv['labels'])) >= 70:
                    raise ValueError('Cluster count exceeds threshold, aborting')
                return idx(indiv, *args, **kwargs)

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
    data, labels = indiv['data'], indiv['labels']
    clusters, centroids = get_clusters_and_centroids(labels, data)
    intra_dists = mean_centroid_distance_separation(clusters=clusters, centroids=centroids)

    centroid_distances = squareform(pdist(centroids))

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists
    scores = np.max(combined_intra_dists / centroid_distances, axis=1)
    return np.mean(scores)


@evaluation_index(minimize=False)
def dvcb_index(indiv: Individual):
    labels = indiv["labels"]
    dists = cache_distances(indiv)
    cluster_validities, cluster_counts = density_based_cluster_validity(dists, labels, indiv['data'].shape[1], return_intcount=True)
    return (cluster_counts * cluster_validities).sum() / len(labels)


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
            matrices = matrices[squareform_matrix(len(cluster_labels)), :, :]
            np.logical_or(matrices, np.swapaxes(matrices, 1, 2), out=matrices)
            matrices = matrices[:, squareform_m]
            dists = np.where(matrices, dists, 0)
            min_distance = (dists.sum(axis=1) / np.count_nonzero(matrices, axis=1)).min()
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

indices = [
    ('silhouette', 'silhouette_index'),
    ('calinski_harabaz', 'calinski_harabaz_index'),
    ('davies_bouldin', 'davies_bouldin_index'),
    ('dvcb', 'dvcb_index'),
    ('dunn', 'dunn_index'),
    ('generalized_dunn_41', 'generalized_dunn_index(separation="centroid_distance", cohesion="diameter")'),
    ('generalized_dunn_43', 'generalized_dunn_index(separation="centroid_distance", cohesion="mean_distance")'),
    ('generalized_dunn_51', 'generalized_dunn_index(separation="mean_per_cluster", cohesion="diameter")'),
    ('generalized_dunn_53', 'generalized_dunn_index(separation="mean_per_cluster", cohesion="mean_distance")'),
    ('generalized_dunn_13', 'generalized_dunn_index(separation="single_linkage", cohesion="mean_distance")'),
    # ('generalized_dunn_31', 'generalized_dunn_index(separation="mean_per_point", cohension="diameter")'),
    # ('generalized_dunn_33', 'generalized_dunn_index(separation="mean_per_point", cohension="mean_distance")'),
]

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
