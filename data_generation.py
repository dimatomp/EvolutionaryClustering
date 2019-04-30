import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import normalize


def generate_random_normal(max_points, dim=None, n_clusters=None):
    dim = dim or np.random.randint(2, 50)
    n_clusters = n_clusters or np.random.randint(2, 50)
    centers = np.random.uniform(low=0, high=50, size=(n_clusters, dim))
    result = []
    for i, center in enumerate(centers):
        n_points = np.random.randint(10, max(11, max_points // n_clusters))
        result.append(np.random.multivariate_normal(center, np.identity(dim), size=n_points))
    return np.vstack(result)


def generated_file_name(max_points, dim=None, n_clusters=None):
    return 'generated_random_normal_{}_{}_{}.csv'.format(max_points, dim, n_clusters)


def load_generated_random_normal(max_points, dim=None, n_clusters=None, prefix='.'):
    fname = prefix + '/' + generated_file_name(max_points, dim, n_clusters)
    datas = np.loadtxt(fname, delimiter=',')
    return datas[:, :-1]  # , datas[:, -1]


def normalize_data(dataset):
    unique = np.unique(dataset, axis=0)
    return normalize(unique, axis=0)


def load_immunotherapy(prefix='.'):
    dataset = pd.read_excel(prefix + '/Immunotherapy.xlsx')
    return dataset.values


def transform_dataset(dataset):
    cols = []
    for i, (col, dtype) in enumerate(zip(dataset.columns, dataset.dtypes)):
        try:
            elems = np.unique(dataset[col])
        except TypeError:
            elems = np.array(list(set(dataset[col])))
        if len(elems) == 1 or len(elems) == len(dataset) and dtype.kind in "iu" and i == 0:
            # it's either a single value or most likely an ID column
            print('Excluding column number', i, 'because it is',
                  'a single value' if len(elems) == 1 else 'most likely an index column', file=sys.stderr)
            continue
        if dtype.kind in "fiu":
            cols.append(dataset[col])
        else:
            elems = np.array(list(sorted(elems)))
            indices = np.argwhere(dataset[col][:, None] == elems[None, :])[:, 1]
            cols.append(indices.flatten())
    return np.array(cols).T


def load_user_knowledge(prefix='.'):
    dataset = pd.read_excel(prefix + '/Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls', sheet_name=1)
    return transform_dataset(dataset[dataset.columns[:-3]])


def load_from_file(fname, prefix='.'):
    return transform_dataset(pd.read_csv(prefix + '/' + fname))


def load_sales_transactions(prefix='.'):
    return load_from_file('Sales_Transactions_Dataset_Weekly.csv', prefix=prefix)[:, 53:]
