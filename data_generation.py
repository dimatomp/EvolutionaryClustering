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


def load_user_knowledge():
    dataset = pd.read_excel('../Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls', sheet_name=1)
    data = dataset.values[:, :-4].astype('float')
    clusters = list(set(dataset.values[:, -4]))
    labels = np.array([clusters.index(i) for i in dataset.values[:, -4]])
    return data, labels


def load_regular_csv(filename, ignore=1):
    dataset = pd.read_csv(filename)
    data = dataset.values[:, :-ignore].astype('float')
    clusters = list(set(dataset.values[:, -ignore]))
    labels = np.array([clusters.index(i) for i in dataset.values[:, -ignore]])
    return data, labels


def load_iris():
    return load_regular_csv('../iris.data')


def load_mfeat_morphological():
    return load_regular_csv('../dataset_18_mfeat-morphological.csv')


def load_glass():
    return load_regular_csv('../dataset_41_glass.csv')


def load_haberman():
    return load_regular_csv('../dataset_43_haberman.csv')


def load_heart_statlog():
    return load_regular_csv('../dataset_53_heart-statlog.csv')


def load_vehicle():
    return load_regular_csv('../dataset_54_vehicle.csv')


def load_liver_disorders():
    return load_regular_csv('../dataset_8_liver-disorders.csv', ignore=2)


def load_oil_spill():
    dataset = pd.read_csv('../phpgEDZ0c.csv')
    labels = dataset['class']
    clusters = list(set(labels))
    del dataset['class']
    labels = np.array([clusters.index(i) for i in labels])
    return dataset.values, labels
