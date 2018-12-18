import numpy as np


def generate_random_normal(max_points, dim=None, n_clusters=None):
    dim = dim or np.random.randint(2, 50)
    n_clusters = n_clusters or np.random.randint(2, 50)
    centers = np.random.uniform(low=0, high=50, size=(n_clusters, dim))
    result = []
    for center in centers:
        n_points = np.random.randint(10, max(11, max_points // n_clusters))
        result.append(np.random.multivariate_normal(center, np.identity(dim), size=n_points))
    return np.vstack(result), n_clusters
