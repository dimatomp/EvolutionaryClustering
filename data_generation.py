import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def generate_random_normal(max_points, dim=None, n_clusters=None):
    dim = dim or np.random.randint(2, 50)
    n_clusters = n_clusters or np.random.randint(2, 50)
    centers = np.random.uniform(low=0, high=50, size=(n_clusters, dim))
    result = []
    clusters = []
    for i, center in enumerate(centers):
        n_points = np.random.randint(10, max(11, max_points // n_clusters))
        result.append(np.random.multivariate_normal(center, np.identity(dim), size=n_points))
        clusters.append(np.full(n_points, i))
    return np.vstack(result), np.hstack(clusters)


def normalize_data(dataset):
    return normalize(dataset[0], axis=0), dataset[1]


def load_immunotherapy():
    dataset = pd.read_excel('../Immunotherapy.xlsx')
    clusters = dataset['Result_of_Treatment']
    del dataset['Result_of_Treatment']
    return dataset.values, clusters.values


def load_iris():
    dataset = pd.read_csv('../iris.data')
    data = dataset.values[:, :-1].astype('float')
    clusters = list(set(dataset.values[:, -1]))
    labels = np.array([clusters.index(i) for i in dataset.values[:, -1]])
    return data, labels

# TODO 8 more real datasets
