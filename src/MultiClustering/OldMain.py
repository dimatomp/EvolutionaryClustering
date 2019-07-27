# import pandas as pd
# from sklearn import datasets
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.cluster import AffinityPropagation
# from sklearn.cluster import MeanShift, estimate_bandwidth
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.cluster import DBSCAN
# from sklearn.model_selection import train_test_split
# import pysmac
# import numpy as np
#
# import Metric
# import Constants
#
# #train_datas = pd.read_csv('aaa.csv', header=0)
# iris = datasets.load_iris()
# X_all = iris.data
# y_all = iris.target
#
# n_samples = 500
#
# # 2 nonoverlap circlues
# #noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
# #X_all, y_all = noisy_circles
#
# # 2 nonoverlap
# #noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
# #X_all, y_all = noisy_moons
#
# # 3 strange
# #X_all, y_all = datasets.make_blobs(n_samples=n_samples, random_state=170)
# #transformation = [[0.4, -0.5], [-0.4, 0.8]]
# #X = np.dot(X_all, transformation)
#
# #varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[0.1, 2.5, 0.5])
# #X_all, y_all = varied
#
# # 3 easily observed clusters
# #X, y = datasets.make_blobs(n_samples=n_samples, n_features=4, centers=3, cluster_std=1, center_box=(-10.0, 10.0), shuffle=True)
#
# #X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25)
# X = X_all
#
# max_eval = 10
# metric = ""
#
# params_km = dict( \
#     n_clusters=('integer', [2, 15], 8),
#     n_init=('integer', [2, 15], 10),
#     max_iter=('integer', [50, 1500], 300),
#     tol=('real', [1e-6, 1e-2], 1e-4),
#     #precompute_distances=('categorical', ["'auto'", "True", "False"], "'auto'"),
#     verbose=('integer', [0, 10], 0),
#     algorithm=('categorical', ["'auto'", "'full'", "'elkan'"], "'auto'"),
# )
#
# params_aff = dict(
#         damping = ('real', [0.5, 1], 0.5),
#         max_iter = ('integer', [100, 1000], 200),
#         convergence_iter = ('integer', [5, 20], 15),
#     )
#
# params_ms = dict(
#     quantile = ('real', [0, 1], 0.3),
#     bin_seeding = ('integer', [0, 1], 0),
#     min_bin_freq = ('integer', [1, 100], 1),
#     cluster_all = ('integer', [0, 1], 0),
#     )
#
# params_w = dict(
#     n_clusters = ('integer', [2, 15], 2),
#     affinity = ('categorical', ["'euclidean'", "'l1'", "'l2'", "'manhattan'", "'cosine'", "'precomputed'"], "'euclidean'"),
#     linkage = ('categorical', ["'ward'", "'complete'", "'average'"],"'ward'"),
# )
#
# params_db = dict(
#     eps = ('real', [0.1, 0.9], 0.5),
#     min_samples = ('integer', [2, 10], 5),
#     algorithm = ('categorical', ["'auto'", "'ball_tree'", "'kd_tree'", "'brute'"], "'auto'"),
#     leaf_size = ('integer', [5, 100], 30),
# )
#
# def run_km(n_clusters, n_init, max_iter, tol, verbose, algorithm):
#     cl = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter,
#                 tol=tol, verbose=verbose, algorithm=algorithm)
#     return run(cl)
#
# def run_aff(damping, max_iter, convergence_iter):
#     cl = AffinityPropagation(damping=damping, max_iter=max_iter, convergence_iter=convergence_iter)
#     return run(cl)
#
# def run_ms(quantile, bin_seeding, min_bin_freq, cluster_all):
#     bandwidth = estimate_bandwidth(X, quantile=quantile)
#     cl = MeanShift(bandwidth=bandwidth, bin_seeding=bool(bin_seeding), min_bin_freq=min_bin_freq, cluster_all=bool(cluster_all))
#     return run(cl)
#
# def run_w(n_clusters, affinity, linkage):
#     cl = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
#     return run(cl)
#
# def run_db(eps, min_samples, algorithm, leaf_size):
#     cl = DBSCAN(eps=eps, min_samples=min_samples, algorithm=algorithm, leaf_size=leaf_size)
#     return run(cl)
#
# def run(cl):
#     cl.fit(X)
#     labels = cl.labels_
#     #centers = cl.cluster_centers_
#     labels_unique = np.unique(labels)
#     n_clusters = len(labels_unique)
#     print("n_clusters = " + str(n_clusters))
#     m = Metric.metric(X, n_clusters, labels, metric)
#     print("metric = " + str(m))
#     return m
#
# best_val = 1.0
# best_algo = "-1"
# best_params = "-1"
#
# i = 0
# algos = {Constants.dbscan_algo:0, Constants.kmeans_algo:0, Constants.affinity_algo:0, Constants.mean_shift_algo:0, Constants.ward_algo:0}
# metrics = [Constants.davies_bouldin_metric, Constants.dunn_metric, Constants.cal_har_metric, Constants.silhouette_metric,
#            Constants.dunn31_metric, Constants.dunn41_metric, Constants.dunn51_metric, Constants.dunn33_metric, Constants.dunn43_metric,
#            Constants.dunn53_metric]
# #metrics = ["sc"]
# saved_parameters = [""] * len(metrics)
# num_parameters_for_algo = {Constants.kmeans_algo:[], Constants.affinity_algo:[], Constants.mean_shift_algo:[], Constants.ward_algo:[], Constants.dbscan_algo:[]}
# for metric in metrics:
#     for algo in algos.keys():
#         opt = pysmac.SMAC_optimizer()
#         value = 1
#         parameters = ""
#         if (Constants.kmeans_algo in algo):
#             value, parameters = opt.minimize(
#                 func=run_km,  # the function to be minimized
#                 max_evaluations=max_eval,  # the number of function calls allowed
#                 parameter_dict=params_km,  # the parameter dictionary
#                 seed=11)
#         elif (Constants.affinity_algo in algo):
#             value, parameters = opt.minimize(
#                 func=run_aff,  # the function to be minimized
#                 max_evaluations=max_eval,  # the number of function calls allowed
#                 parameter_dict=params_aff,  # the parameter dictionary
#                 seed=11)
#         elif (Constants.mean_shift_algo in algo):
#             value, parameters = opt.minimize(
#                 func=run_ms,  # the function to be minimized
#                 max_evaluations=max_eval,  # the number of function calls allowed
#                 parameter_dict=params_ms,  # the parameter dictionary
#                 seed=11)
#         elif (Constants.ward_algo in algo):
#             value, parameters = opt.minimize(
#                 func=run_w,  # the function to be minimized
#                 max_evaluations=max_eval,  # the number of function calls allowed
#                 parameter_dict=params_w,  # the parameter dictionary
#                 seed=11)
#         elif (Constants.dbscan_algo in algo):
#             value, parameters = opt.minimize(
#                 func=run_db,  # the function to be minimized
#                 max_evaluations=max_eval,  # the number of function calls allowed
#                 parameter_dict=params_db,  # the parameter dictionary
#                 seed=11)
#         # the return value is a tuple of the lowest function value and a dictionary
#         # containing corresponding parameter setting.
#         print(('For algo ' + algo + ' lowest function value found: %f' % value))
#         print(('Parameter setting %s' % parameters))
#         if (value < best_val):
#             best_val = value
#             best_algo = algo
#             best_params = parameters
#     algos[best_algo] += 1
#     saved_parameters[i] = best_params
#     num_parameters_for_algo[best_algo].append(i)
#     i += 1
#
# chosen_algo = ""
# num_cases = 0
# for algo in algos.keys():
#     if (algos[algo] > num_cases):
#         num_cases = algos[algo]
#         chosen_algo = algo
#
# best_params = saved_parameters[num_parameters_for_algo[chosen_algo][0]]
# cl = ""
# if   (Constants.kmeans_algo in chosen_algo):
#     cl = KMeans(n_clusters=best_params["n_clusters"], n_init=best_params["n_init"], max_iter=best_params["max_iter"],
#                 tol=best_params["tol"], verbose=best_params["verbose"], algorithm=best_params["algorithm"])
# elif (Constants.affinity_algo in chosen_algo):
#     cl = AffinityPropagation(damping=best_params["damping"], max_iter=best_params["max_iter"], convergence_iter=best_params["convergence_iter"])
# elif (Constants.mean_shift_algo in chosen_algo):
#     bandwidth = estimate_bandwidth(X, quantile=best_params["quantile"])
#     cl = MeanShift(bandwidth=bandwidth, bin_seeding=bool(best_params["bin_seeding"]), min_bin_freq=best_params["min_bin_freq"],
#               cluster_all=bool(best_params["cluster_all"]))
# elif (Constants.ward_algo in chosen_algo):
#     cl = AgglomerativeClustering(n_clusters=best_params["n_clusters"], affinity=best_params["affinity"], linkage=best_params["linkage"])
# elif (Constants.dbscan_algo in chosen_algo):
#     cl = DBSCAN(eps=best_params["eps"], min_samples=best_params["min_samples"], algorithm=best_params["algorithm"], leaf_size=best_params["leaf_size"])
#
# cl.fit(X)
# plt.figure(figsize=(15, 10))
# plt.subplot(221)
# plt.scatter(X[:, 0], X[:, 3], c=cl.labels_)
# plt.title("Predicted lables")
#
# plt.subplot(222)
# plt.scatter(X[:, 0], X[:, 3], c=y_all)
# plt.title("True lables")
#
# print('Best algorithm = ' + chosen_algo)
# #print('Best metric = ' + str(best_val))
# print('Best parameters = ' + str(best_params))
# print(str(algos))
# #print(algo.score(X_test, y_test))
# plt.show()