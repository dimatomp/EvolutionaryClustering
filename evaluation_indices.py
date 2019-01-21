import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, calinski_harabaz_score, davies_bouldin_score


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


def generalized_dunn_index(separation, cohension):
    @evaluation_index(minimize=False)
    def index(indiv):
        labels, data = indiv["labels"], indiv["data"]
        cluster_labels = np.unique(labels)
        clusters = [data[labels == i] for i in cluster_labels]
        centroids = np.array([d.mean(axis=0) for d in clusters])

        def cache_distances():
            if "distances" in indiv:
                distances = indiv["distances"]
            else:
                distances = pdist(data)
                indiv["distances"] = distances
            return distances

        if separation == "single_linkage":
            dists = cache_distances()
            min_distance = dists[squareform(labels[:, None] != labels[None, :])].min()
        elif separation == "mean_per_point":
            # TODO Too slow
            dists = cache_distances()
            min_distance = np.inf
            matrices = [labels[:, None] == i for i in cluster_labels]
            for i in range(len(cluster_labels) - 1):
                for j in range(i + 1, len(cluster_labels)):
                    matrix = matrices[i] & matrices[j].T
                    min_distance = min(min_distance, dists[squareform(matrix | matrix.T)].mean())
        elif separation == "centroid_distance":
            min_distance = pdist(centroids).min()
        elif separation == "mean_per_cluster":
            min_distance = min(min(
                norm(cluster1 - centroid2).sum() + norm(cluster2 - centroid1).sum() / (len(cluster1) + len(cluster2))
                for
                cluster2, centroid2 in zip(clusters[idx + 1:], centroids[idx + 1:])) for idx, (cluster1, centroid1) in
                               enumerate(zip(clusters[:-1], centroids[:-1])))
        if cohension == "diameter":
            points = [d[np.argmax(norm(d - c, axis=1))] for d, c in zip(clusters, centroids)]
            max_diameter = max(norm(d - p, axis=1).max() for d, p in zip(clusters, points))
        elif cohension == "mean_distance":
            max_diameter = 2 * max(norm(cluster - centroid).mean() for cluster, centroid in zip(clusters, centroids))
        return min_distance / max_diameter

    return index


dunn_index = generalized_dunn_index(separation="single_linkage", cohension="diameter")

# TODO 3 more (density-based?) measures, not from [2009]

'''
class DynamicIndex:
    def __init__(self, index):
        self.index = index
        self.prev_labels = None

    def __call__(self, indiv):
        labels, data = indiv
        n_clusters = len(np.unique(labels))
        if self.prev_labels is None:
            self.result = self.index.find(data, labels, n_clusters)
        else:
            for idx in np.argwhere(self.prev_labels != labels).flatten():
                self.result = self.index.update(data, n_clusters, labels, self.prev_labels[idx], labels[idx], idx)
        self.prev_labels = labels
        return self.result


class DynamicVRCIndex(DynamicIndex):
    def __init__(self):
        super(DynamicVRCIndex, self).__init__(VRCIndex())


class DynamicSilhouetteIndex(DynamicIndex):
    def __init__(self):
        super(DynamicSilhouetteIndex, self).__init__(SilhouetteIndex())
'''
