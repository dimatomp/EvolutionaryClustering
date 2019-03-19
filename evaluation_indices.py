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


def dvcb_index(d=2):
    @evaluation_index(minimize=False)
    def index(indiv):
        labels = indiv["labels"]
        dists = cache_distances(indiv)
        cluster_counts = np.bincount(labels)
        cluster_validities = density_based_cluster_validity(dists, labels, d=d)
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
