from sklearn import datasets
import numpy as np

from . import Constants

def fill_data_iris():
    iris = datasets.load_iris()
    X_all = iris.data
    y_all = iris.target
    return X_all, y_all

    # 2 nonoverlap circlues
def fill_data_circles():
    noisy_circles = datasets.make_circles(n_samples=Constants.n_samples, factor=0.5, noise=0.05)
    X_all, y_all = noisy_circles
    return X_all, y_all

    # 2 nonoverlap
def fill_data_moons():
    noisy_moons = datasets.make_moons(n_samples=Constants.n_samples, noise=0.05)
    X_all, y_all = noisy_moons
    return X_all, y_all

    # 3 strange
def fill_data_3():
    X_all, y_all = datasets.make_blobs(n_samples=Constants.n_samples, random_state=170)
    transformation = [[0.4, -0.5], [-0.4, 0.8]]
    X_all = np.dot(X_all, transformation)
    return X_all, y_all

def fill_data_2():
    varied = datasets.make_blobs(n_samples=Constants.n_samples, cluster_std=[0.1, 2.5, 0.5])
    X_all, y_all = varied
    return X_all, y_all

    # 3 easily observed clusters
def fill_data_easy():
    X, y = datasets.make_blobs(n_samples=Constants.n_samples, n_features=4, centers=3, cluster_std=1, center_box=(-10.0, 10.0), shuffle=True)
    return X, y