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


def generated_file_name(max_points, dim=None, n_clusters=None):
    return 'generated_random_normal_{}_{}_{}.csv'.format(max_points, dim, n_clusters)


def load_generated_random_normal(max_points, dim=None, n_clusters=None, prefix='.'):
    fname = prefix + '/' + generated_file_name(max_points, dim, n_clusters)
    datas = np.loadtxt(fname, delimiter=',')
    return datas[:, :-1], datas[:, -1]


def normalize_data(dataset):
    return normalize(dataset[0], axis=0), dataset[1]


def load_immunotherapy(prefix='.'):
    dataset = pd.read_excel(prefix + '/Immunotherapy.xlsx')
    clusters = dataset['Result_of_Treatment']
    del dataset['Result_of_Treatment']
    return dataset.values, clusters.values


def load_user_knowledge(prefix='.'):
    dataset = pd.read_excel(prefix + '/Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls', sheet_name=1)
    data = dataset.values[:, :-4].astype('float')
    clusters = list(set(dataset.values[:, -4]))
    labels = np.array([clusters.index(i) for i in dataset.values[:, -4]])
    return data, labels


def load_regular_csv(filename, ignore=1, prefix='.', heading=True):
    dataset = pd.read_csv(prefix + '/' + filename)
    data = dataset.values[:, :-ignore].astype('float')
    clusters = list(set(dataset.values[:, -ignore]))
    labels = np.array([clusters.index(i) for i in dataset.values[:, -ignore]])
    return data, labels


def load_iris(prefix='.'):
    return load_regular_csv('iris.data', prefix=prefix)


def load_mfeat_morphological(prefix='.'):
    return load_regular_csv('dataset_18_mfeat-morphological.csv', prefix=prefix)


def load_glass(prefix='.'):
    return load_regular_csv('dataset_41_glass.csv', prefix=prefix)


def load_haberman(prefix='.'):
    return load_regular_csv('dataset_43_haberman.csv', prefix=prefix)


def load_heart_statlog(prefix='.'):
    return load_regular_csv('dataset_53_heart-statlog.csv', prefix=prefix)


def load_vehicle(prefix='.'):
    return load_regular_csv('dataset_54_vehicle.csv', prefix=prefix)


def load_liver_disorders(prefix='.'):
    return load_regular_csv('dataset_8_liver-disorders.csv', ignore=2, prefix=prefix)


def load_oil_spill(prefix='.'):
    dataset = pd.read_csv(prefix + '/phpgEDZOc.csv')
    labels = dataset['class']
    clusters = list(set(labels))
    del dataset['class']
    labels = np.array([clusters.index(i) for i in labels])
    return dataset.values, labels
