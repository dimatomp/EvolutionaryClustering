from .cluster_measures import *
from .individual import Individual
from scipy.sparse import find, coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components


def initializer(initfunc):
    def init(*args, **kwargs):
        dictionary = initfunc(*args, **kwargs)
        result = Individual({'data': dictionary['data']})
        for key, value in dictionary.items():
            if key != 'data':
                result.set_partition_field(key, value)
        return result

    return init


@initializer
def random_initialization(data, n_clusters):
    return {"labels": np.random.randint(low=0, high=n_clusters, size=len(data)), "data": data}


def tree_initialization(data, n_clusters):
    indiv = Individual({"data": data})
    rows, cols, vals = find(minimum_spanning_tree(squareform(cache_distances(indiv))))
    indiv.set_data_field("mst", (rows, cols, vals))
    perm = np.argsort(vals)
    edges = perm[:-n_clusters+1]
    mst = coo_matrix((vals[edges], (rows[edges], cols[edges])), shape=(len(data), len(data)))
    labels = connected_components(mst)[1]
    split_edges = np.zeros(len(rows), dtype='bool')
    split_edges[perm[-n_clusters+1:]] = True
    indiv.set_partition_field('labels', labels)
    indiv.set_partition_field('split_edges', split_edges)
    return indiv


@initializer
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


@initializer
def centroid_initialization(data, n_clusters):
    datamin, datamax = data.min(axis=0), data.max(axis=0)
    centroids = np.random.sample((n_clusters, data.shape[1])) * (datamax - datamin) + datamin
    labels, centroids = get_labels_by_centroids(centroids, data)
    return {"labels": labels, "data": data, "centroids": centroids}


@initializer
def prototype_initialization(data, n_clusters):
    prototypes = np.zeros(len(data), dtype='bool')
    prototypes[np.random.choice(len(data), n_clusters)] = True
    return {"labels": get_labels_by_prototypes(prototypes, data), "prototypes": prototypes, "data": data}
