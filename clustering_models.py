import numpy as np
from scipy.spatial.distance import pdist, squareform


class IntegerEncodingModel: pass


class RandomInitModel(IntegerEncodingModel):
    def generate_initial(self, data, n_clusters):
        return np.random.randint(low=0, high=n_clusters, size=len(data)), data


class OneNthChangeModel(RandomInitModel):
    def mutate(self, indiv):
        values, data = indiv
        numbers_to_change = np.zeros(len(values), dtype='bool')
        numbers_to_change[np.random.randint(len(values))] = 1
        numbers_to_change[np.random.randint(len(values), size=len(values)) == 0] |= True
        cluster_sizes = np.bincount(values)
        n_clusters = len(cluster_sizes)
        if np.random.randint(n_clusters ** 2) == 0:
            n_clusters += 1
        values = values.copy()
        values[numbers_to_change] = np.random.randint(n_clusters, size=np.count_nonzero(numbers_to_change))
        empty = np.cumsum(np.bincount(values) == 0)
        values -= empty[values]
        return values, data


class AngleInitModel(IntegerEncodingModel):
    def generate_initial(self, data, n_clusters):
        centroid = data.mean(axis=0)
        norm = np.random.multivariate_normal(np.zeros(len(centroid)), np.identity(len(centroid)))
        dots = (data - centroid).dot(norm)
        dotmin = dots.min()
        labels = np.minimum(np.floor((dots - dotmin) / (dots.max() - dotmin) * n_clusters).astype('int'),
                            n_clusters - 1)
        emptyLabels = np.cumsum(np.bincount(labels) == 0)
        labels -= emptyLabels[labels]
        return labels, data


class SplitMergeMoveModel(AngleInitModel):
    def generate_initial(self, data, n_clusters):
        return super(SplitMergeMoveModel, self).generate_initial(data, 2)

    def mutate(self, indiv):
        values, data = indiv
        cluster_sizes = np.bincount(values)
        values = values.copy()
        while True:
            method = np.random.randint(3)
            if method == 0:
                ex_cluster = np.random.choice(np.argwhere(cluster_sizes > 1).flatten())
                centroid = data[values == ex_cluster].mean(axis=0)
                norm = np.random.multivariate_normal(np.zeros(len(centroid)), np.identity(len(centroid)))
                dots = (data[values == ex_cluster] - centroid).dot(norm)
                ratio = np.random.uniform(dots.min(), dots.max())
                negativeDot = dots < ratio
                negativeSum = np.count_nonzero(negativeDot)
                if negativeSum == 0 or negativeSum == len(negativeDot):
                    continue
                values[values == ex_cluster] = np.where(negativeDot, len(cluster_sizes), ex_cluster)
            elif method == 1 and len(cluster_sizes) != 2:
                centroids = np.array([data[values == i].mean(axis=0) for i in range(len(cluster_sizes))])
                dists = pdist(centroids, 'minkowski', p=1)
                dists = np.exp(-dists - np.log(np.exp(-dists).sum()))
                # dists = dists.max() - dists
                dists /= dists.sum()
                pair = np.random.choice(len(dists), p=dists)
                dists = np.zeros(len(dists))
                dists[pair] = 1
                src_cluster, dst_cluster = np.argwhere(squareform(dists) == 1)[0]
                # src_cluster, dst_cluster = np.random.choice(len(cluster_sizes), 2, replace=False)
                values[values == src_cluster] = dst_cluster
                values[values > src_cluster] -= 1
            else:
                # src_cluster = np.random.randint(len(cluster_sizes))
                dst_cluster = np.random.randint(len(cluster_sizes))  # - 1)
                # if dst_cluster >= src_cluster:
                #    dst_cluster += 1
                centroid = data[values == dst_cluster].mean(axis=0)
                otherElems = values != dst_cluster  # == src_cluster
                dists = np.linalg.norm(data[otherElems] - centroid, ord=1, axis=1)
                # dists = dists.max() - dists
                dists = np.exp(-dists - np.log(np.exp(-dists).sum()))
                dists /= dists.sum()
                n_points = np.count_nonzero(dists)
                n_points = np.random.binomial(n_points - 1, 1 / (n_points - 1)) + 1
                n_indices = np.random.choice(np.argwhere(otherElems).flatten(), n_points, replace=False, p=dists)
                values[n_indices] = dst_cluster
                emptyLabels = np.cumsum(np.bincount(values) == 0)
                values -= emptyLabels[values]
            break
        return values, data
