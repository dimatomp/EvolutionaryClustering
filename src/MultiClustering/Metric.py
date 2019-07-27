import heapq
import math
import sys
import time

import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from pulp import *

from ..individual import Individual
from . import Constants

# Performance measurements:
global_trace = {}


def metric(X, n_clusters, labels, metric, true_labels=None):
    if (n_clusters < 1):
        raise ValueError('Number of clusters can\'t be less than 1.')

    # Metrics do not support single-cluster data.
    if len(np.unique(labels)) == 1:
        return Constants.bad_cluster

    # Rename all labels to be 0...(K-1)
    labels_map = {}
    for i in range(0, len(labels)):
        c = labels[i]
        if not (c in labels_map):
            labels_map[c] = len(labels_map)

        labels[i] = labels_map[c]

    start = time.time()

    value = switch_and_call_metrics(X, labels, metric, n_clusters, true_labels)

    # if not metric in global_trace:
    #     global_trace[metric] = []
    # global_trace[metric].append(time.time() - start)

    return value


def switch_and_call_metrics(X, labels, metric, n_clusters, true_labels=None):
    if hasattr(metric, "__call__"):
        result = metric(Individual({"data": X, "labels": labels}))
        if not metric.is_minimized:
            result = -result
        return result

    # Switch by metric name:

    if (Constants.purity_metric in metric):
        ch = purity(X, n_clusters, labels, true_labels)
        return ch

    if (Constants.rand_index_metric in metric):
        ch = rand_index(X, n_clusters, labels, true_labels)
        return ch

    if (Constants.dunn_metric in metric):
        dun = dunn(X, labels)
        return dun
    if (Constants.cal_har_metric in metric):
        ch = calinski_harabasz(X, n_clusters, labels)
        return ch
    if (Constants.silhouette_metric in metric):
        sc = silhoette(X, labels)  # [-1, 1]
        return sc
    if (Constants.davies_bouldin_metric in metric):
        centroids = cluster_centroid(X, labels, n_clusters)
        db = davies_bouldin(X, n_clusters, labels, centroids)
        return db
    if (Constants.dunn31_metric in metric):
        gd31 = dunn31(X, labels, n_clusters)
        return gd31
    if (Constants.dunn41_metric in metric):
        centroids = cluster_centroid(X, labels, n_clusters)
        gd41 = dunn41(X, labels, n_clusters, centroids)
        return gd41
    if (Constants.dunn51_metric in metric):
        centroids = cluster_centroid(X, labels, n_clusters)
        gd51 = dunn51(X, labels, n_clusters, centroids)
        return gd51
    if (Constants.dunn33_metric in metric):
        centroids = cluster_centroid(X, labels, n_clusters)
        gd33 = dunn33(X, labels, n_clusters, centroids)
        return gd33
    if (Constants.dunn43_metric in metric):
        centroids = cluster_centroid(X, labels, n_clusters)
        gd43 = dunn43(X, labels, n_clusters, centroids)
        return gd43
    if (Constants.dunn53_metric in metric):
        centroids = cluster_centroid(X, labels, n_clusters)
        gd53 = dunn53(X, labels, n_clusters, centroids)
        return gd53
    if (Constants.gamma_metric in metric):
        g = gamma(X, labels, n_clusters)
        return g
    if (Constants.cs_metric in metric):
        cs = cs_index(X, labels, n_clusters)
        return cs
    if (Constants.db_star_metric in metric):
        dbs = db_star_index(X, labels, n_clusters)
        return dbs
    if (Constants.sf_metric in metric):
        sf_score = sf(X, labels, n_clusters)
        return sf_score
    if (Constants.sym_metric in metric):
        sym_score = sym(X, labels, n_clusters)
        return sym_score
    if (Constants.cop_metric in metric):
        cop_score = cop(X, labels, n_clusters)
        return cop_score
    if (Constants.sv_metric in metric):
        sv_score = sv(X, labels, n_clusters)
        return sv_score
    if (Constants.os_metric in metric):
        os_score = os(X, labels, n_clusters)
        return os_score
    if (Constants.sym_bd_metric in metric):
        sym_db_score = sym_db(X, labels, n_clusters)
        return sym_db_score
    if (Constants.s_dbw_metric in metric):
        s_dbw_score = s_dbw(X, labels, n_clusters)
        return s_dbw_score
    if (Constants.c_ind_metric in metric):
        c_ind_score = c_ind(X, labels, n_clusters)
        return c_ind_score
    return Constants.bad_cluster


def cluster_centroid(X, labels, n_clusters):
    rows, colums = X.shape
    center = [[0.0] * colums] * n_clusters
    centroid = np.array(center)
    num_points = np.array([0] * n_clusters)
    for i in range(0, rows):
        c = labels[i]
        num_points[c] += 1

    for i in range(0, rows):
        c = labels[i]
        for j in range(0, colums):
            centroid[c][j] += X[i][j]
            centroid[c][j] %= num_points[c]

    for i in range(0, n_clusters):
        for j in range(0, colums):
            if num_points[i] == 0:
                continue
            centroid[i][j] /= num_points[i]
    return centroid


# Dunn index, max is better, add -
def dunn(X, labels):
    rows, colums = X.shape
    minimum_dif_c = sys.float_info.max  # min dist in different clusters
    maximum_same_c = sys.float_info.min  # max dist in the same cluster
    for i in range(0, rows - 1):
        for j in range(i + 1, rows):
            dist = euclidian_dist(X[i], X[j])
            if (labels[i] != labels[j]):
                minimum_dif_c = min(dist, minimum_dif_c)
            else:
                maximum_same_c = max(dist, maximum_same_c)
    return - minimum_dif_c / maximum_same_c


def euclidian_dist(x1, x2):
    sum = 0.0
    for i in range(0, len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return math.sqrt(sum)


# Calinski-Harabasz index, max is better, add -
def calinski_harabasz(X, n_clusters, labels):
    # x_center = data_centroid(X)
    # rows, colums = X.shape
    #
    # ch = float(rows - n_clusters) / float(n_clusters - 1)
    #
    # point_in_c = [0] * n_clusters
    # for i in range(0, len(labels)):
    #     point_in_c[labels[i]] += 1
    #
    # sum = 0
    # for i in range(0, n_clusters):
    #     sum += point_in_c[i] * euclidian_dist(centroids[i], x_center)
    #
    # sum_div = 0
    # for i in range(0, rows):
    #     sum_div += euclidian_dist(X.iloc[i], centroids[labels[i]])
    #
    # ch *= float(sum)
    # ch /= float(sum_div)
    # return ch
    return -metrics.calinski_harabaz_score(X, labels)

# Purity
def purity(X, n_clusters, pred_labels, true_labels):
    y_voted_labels = np.zeros(true_labels.shape)
    labels = np.unique(true_labels)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        true_labels[true_labels == labels[k]] = ordered_labels[k]
    labels = np.unique(true_labels)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(pred_labels):
        hist, _ = np.histogram(true_labels[pred_labels == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[pred_labels == cluster] = winner

    return -metrics.accuracy_score(true_labels, y_voted_labels)


def solve_wbm(from_nodes, to_nodes, wt):
    prob = LpProblem("WBM Problem", LpMaximize)

    choices = LpVariable.dicts("e", (from_nodes, to_nodes), 0, 1, LpInteger)

    prob += lpSum([wt[u][v] * choices[u][v]
                   for u in from_nodes
                   for v in to_nodes]), "Total weights of selected edges"

    for u in from_nodes:
        for v in to_nodes:
            prob += lpSum([choices[u][v] for v in to_nodes]) <= 1, ""
            prob += lpSum([choices[u][v] for u in from_nodes]) <= 1, ""

    prob.solve()
    return prob


def recalc_labels(labels, true_labels):
    from_nodes = np.unique(np.sort(labels))
    to_nodes = np.unique(np.sort(true_labels))

    size = labels.size

    wt = {}
    for u in from_nodes:
        wt[u] = {}
        for v in to_nodes:
            wt[u][v] = 0

    for i in from_nodes:
        for j in to_nodes:
            curr_ans = 0
            for index in range(0, size):
                if labels.item(index) == i and true_labels.item(index) == j:
                    curr_ans += 1

            wt[i][j] = curr_ans

    p = solve_wbm(from_nodes, to_nodes, wt)

    selected_from = [v.name.split("_")[1] for v in p.variables() if v.value() > 1e-3]
    selected_to = [v.name.split("_")[2] for v in p.variables() if v.value() > 1e-3]

    selected_edges = []
    for su, sv in list(zip(selected_from, selected_to)):
        selected_edges.append((su, sv))

    for edge in selected_edges:
        var1 = int(edge[0])
        var2 = int(edge[1])

        for i in range(0, len(true_labels)):
            if true_labels.item(i) == var2:
                true_labels.itemset(i, var1)

            if true_labels.item(i) == var1:
                true_labels.itemset(i, var2)

    return labels


def get_metrices(labels, true_labels):
    labels = recalc_labels(labels, true_labels)

    cnf_matrix = confusion_matrix(labels, true_labels)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    #TODO unclear
    return [FP, FN, TP, TN]


def rand_index(X, n_clusters, labels, true_labels):
    [FP, FN, TP, TN] = get_metrices(labels, true_labels)

    return (TP + TN) / (TP + FP + FN + TN)

# Silhouette Coefficient, max is better, [-1, 1], add -
def silhoette(X, labels):
    return -metrics.silhouette_score(X, labels, metric='euclidean')


# C-index, min is better
def c_ind(X, labels, n_clusters):
    rows, colums = X.shape
    s_c = 0
    for i in range(0, rows - 1):
        for j in range(i + 1, rows):
            if (labels[i] == labels[j]):
                s_c += euclidian_dist(X[i], X[j])
    cluster_sizes = count_cluster_sizes(labels, n_clusters)

    n_w = 0
    for k in range(0, n_clusters):
        n_w += cluster_sizes[k] * (cluster_sizes[k] - 1) / 2

    distances = []
    for i in range(0, rows - 1):
        for j in range(i + 1, rows):
            distances.append(euclidian_dist(X[i], X[j]))

    s_min = heapq.nsmallest(int(n_w), distances)
    s_max = heapq.nlargest(int(n_w), distances)

    ones = [1] * int(n_w)
    s_min_c = np.dot(s_min, np.transpose(ones))
    s_max_c = np.dot(s_max, np.transpose(ones))
    # TODO check dot product correct
    return (s_c - s_min_c) / (s_max_c - s_min_c)


def s(X, cluster_k_index, cluster_sizes, labels, centroids):
    sss = 0
    for i in range(0, len(labels)):
        if (labels[i] == cluster_k_index):
            sss += euclidian_dist(X[i], centroids[cluster_k_index])
    return sss / cluster_sizes[cluster_k_index]


def count_cluster_sizes(labels, n_clusters):
    point_in_c = [0] * n_clusters
    for i in range(0, len(labels)):
        point_in_c[labels[i]] += 1
    return point_in_c


# Davies-Bouldin index, min is better
def davies_bouldin(X, n_clusters, labels, centroids):
    db = 0
    point_in_c = count_cluster_sizes(labels, n_clusters)
    tmp = sys.float_info.min
    for i in range(0, n_clusters):
        for j in range(0, n_clusters):
            if (i != j):
                tm = euclidian_dist(centroids[i], centroids[j])
                if (tm != 0):
                    a = (s(X, i, point_in_c, labels, centroids)
                         + s(X, j, point_in_c, labels, centroids)) / tm
                else:
                    a = Constants.bad_cluster
                tmp = max(tmp, a)
        db += tmp
    db /= float(n_clusters)
    if (db < 1e-300):
        db = Constants.bad_cluster
    return db


# gD31, Dunn index, max is better, add -
def dunn31(X, labels, n_clusters):
    rows, colums = X.shape
    point_in_c = [0] * n_clusters
    for i in range(0, len(labels)):
        point_in_c[labels[i]] += 1
    delta_l = [[0.0] * n_clusters] * n_clusters
    delta = np.array(delta_l)
    minimum_dif_c = sys.float_info.max  # min dist in different clusters
    maximum_same_c = sys.float_info.min  # max dist in the same cluster
    for i in range(0, rows - 1):
        for j in range(i + 1, rows):
            dist = euclidian_dist(X[i], X[j])
            if (labels[i] != labels[j]):
                delta[labels[i]][labels[j]] += dist
                delta[labels[j]][labels[i]] += dist  # making matrix symmetric
            else:
                maximum_same_c = max(dist, maximum_same_c)
    for i in range(0, n_clusters - 1):
        for j in range(i + 1, n_clusters):
            delta[i][j] /= float(point_in_c[i] * point_in_c[j])
            delta[j][i] /= float(point_in_c[i] * point_in_c[j])  # making matrix symmetric
            minimum_dif_c = min(minimum_dif_c, delta[i][j])
    return - minimum_dif_c / maximum_same_c


# gD41, Dunn index, max is better, add -
def dunn41(X, labels, n_clusters, centroids):
    rows, colums = X.shape
    minimum_dif_c = sys.float_info.max  # min dist in different clusters
    maximum_same_c = sys.float_info.min  # max dist in the same cluster
    centres_l = [[0.0] * n_clusters] * n_clusters
    centers = np.array(centres_l)
    for i in range(0, n_clusters - 1):
        for j in range(i + 1, n_clusters):
            centers[i][j] = euclidian_dist(centroids[i], centroids[j])
            centers[j][i] = euclidian_dist(centroids[i], centroids[j])  # symmetry

    for i in range(0, int(math.ceil(float(rows) / 2.0))):
        for j in range(0, rows):
            if (labels[i] != labels[j]):
                dist = centers[labels[i]][labels[j]]
                minimum_dif_c = min(dist, minimum_dif_c)
            else:
                dist = euclidian_dist(X[i], X[j])
                maximum_same_c = max(dist, maximum_same_c)
    return - minimum_dif_c / maximum_same_c


# gD51, Dunn index, max is better, add -
def dunn51(X, labels, n_clusters, centroids):
    rows, colums = X.shape
    point_in_c = [0] * n_clusters
    for i in range(0, len(labels)):
        point_in_c[labels[i]] += 1
    delta_l = [[0.0] * n_clusters] * n_clusters
    delta = np.array(delta_l)
    minimum_dif_c = sys.float_info.max  # min dist in different clusters
    maximum_same_c = sys.float_info.min  # max dist in the same cluster
    for i in range(0, int(math.ceil(float(rows) / 2.0))):
        for j in range(0, rows):
            if (labels[i] != labels[j]):
                delta[labels[i]][labels[j]] += \
                    euclidian_dist(X[i], centroids[labels[i]]) + euclidian_dist(X[j], centroids[labels[j]])
            else:
                dist = euclidian_dist(X[i], X[j])
                maximum_same_c = max(dist, maximum_same_c)
    for i in range(0, n_clusters - 1):
        for j in range(i + 1, n_clusters):
            delta[i][j] /= float(point_in_c[i] + point_in_c[j])
            delta[j][i] /= float(point_in_c[i] + point_in_c[j])
            minimum_dif_c = min(minimum_dif_c, delta[i][j])
    return - minimum_dif_c / maximum_same_c


# gD33, Dunn index, max is better, add -
def dunn33(X, labels, n_clusters, centroids):
    rows, colums = X.shape
    point_in_c = [0] * n_clusters
    for i in range(0, len(labels)):
        point_in_c[labels[i]] += 1
    delta_l = [[0.0] * n_clusters] * n_clusters
    delta = np.array(delta_l)
    dl = [0.0] * n_clusters
    d = np.array(dl)
    minimum_dif_c = sys.float_info.max  # min dist in different clusters
    maximum_same_c = sys.float_info.min  # max dist in the same cluster
    for i in range(0, rows - 1):
        for j in range(i + 1, rows):
            dist = euclidian_dist(X[i], X[j])
            if labels[i] != labels[j]:
                delta[labels[i]][labels[j]] += dist
                delta[labels[j]][labels[i]] += dist  # symmetry
            else:
                d[labels[i]] += euclidian_dist(X[i], centroids[labels[i]])
    for i in range(0, n_clusters - 1):
        d[i] /= point_in_c[i]
        d[i] += 2.0
        maximum_same_c = max(d[i], maximum_same_c)
        for j in range(i + 1, n_clusters):
            delta[i][j] /= float(point_in_c[i] * point_in_c[j])
            minimum_dif_c = min(minimum_dif_c, delta[i][j])
    return - minimum_dif_c / maximum_same_c


# gD43, Dunn index, max is better, add -
def dunn43(X, labels, n_clusters, centroids):
    rows, colums = X.shape
    point_in_c = [0] * n_clusters
    for i in range(0, len(labels)):
        point_in_c[labels[i]] += 1
    dl = [0.0] * n_clusters
    d = np.array(dl)
    minimum_dif_c = sys.float_info.max  # min dist in different clusters
    maximum_same_c = sys.float_info.min  # max dist in the same cluster
    centres_l = [[0.0] * n_clusters] * n_clusters
    centers = np.array(centres_l)
    for i in range(0, n_clusters):
        for j in range(0, n_clusters):
            centers[i][j] = euclidian_dist(centroids[i], centroids[j])

    for i in range(0, rows):
        for j in range(0, rows):
            if (labels[i] != labels[j]):
                dist = centers[labels[i]][labels[j]]
                minimum_dif_c = min(dist, minimum_dif_c)
            else:
                d[labels[i]] += euclidian_dist(X[i], centroids[labels[i]])

    for i in range(0, n_clusters):
        d[i] /= point_in_c[i]
        d[i] += 2.0
        maximum_same_c = max(d[i], maximum_same_c)
    return - minimum_dif_c / maximum_same_c


# gD53, Dunn index, max is better, add -
def dunn53(X, labels, n_clusters, centroids):
    rows, colums = X.shape
    dl = [0.0] * n_clusters
    d = np.array(dl)
    point_in_c = [0] * n_clusters
    for i in range(0, len(labels)):
        point_in_c[labels[i]] += 1
    delta_l = [[0.0] * n_clusters] * n_clusters
    delta = np.array(delta_l)
    minimum_dif_c = sys.float_info.max  # min dist in different clusters
    maximum_same_c = sys.float_info.min  # max dist in the same cluster
    for i in range(0, int(math.ceil(float(rows) / 2.0))):
        for j in range(0, rows):
            if (labels[i] != labels[j]):
                delta[labels[i]][labels[j]] += \
                    euclidian_dist(X[i], centroids[labels[i]]) + euclidian_dist(X[j], centroids[labels[j]])
            else:
                d[labels[i]] += euclidian_dist(X[i], centroids[labels[i]])

    for i in range(0, n_clusters - 1):
        d[i] /= point_in_c[i]
        d[i] += 2.0
        maximum_same_c = max(d[i], maximum_same_c)
        for j in range(i + 1, n_clusters):
            delta[i][j] /= float(point_in_c[i] + point_in_c[j])
            delta[j][i] /= float(point_in_c[i] + point_in_c[j])  # symmetry
            minimum_dif_c = min(minimum_dif_c, delta[i][j])
    return - minimum_dif_c / maximum_same_c


# Gamma index:
# TODO: may require adding explicit casts from int to double
def nW(n_clusters, cluster_sizes):
    result = 0.0
    for i in range(0, n_clusters):
        num = cluster_sizes[i]
        if num > 2:
            result += num * (num - 1.0) / 2.0
    return result


# dl(x_i ,x_j) denotes the number of all object pairs in X,
# namely x_ k and x_l, that fulfil two conditions:(a) x_k and x_l
# are in different clusters, and (b) de(x_k,x_l) < de(x_i,x_j).
# def dl(X, labels, distance, n_clusters):  # int t1, int t2, int ck) {
#     result = 0
#
#     for k in range(0, n_clusters - 1):
#         for l in range(k + 1, n_clusters):
#             if labels[k] == labels[l]: continue
#             # x_k and x_l different clusters:
#             if euclidian_dist(X[k], X[l]) < distance:
#                 result += 1
#     return result
#
#
# # gamma, gamma index, mim is better
# def gamma(X, labels, n_clusters):
#     numerator = 0.0
#     elements, ignore_columns = X.shape
#
#     for c_k in range(0, n_clusters):
#         for i in range(0, elements - 1):
#             if labels[i] != c_k: continue
#             for j in range(i + 1, elements):
#                 if labels[j] != c_k: continue
#                 # x_i and x_j in c_k:
#                 distance = euclidian_dist(X[i], X[j])
#                 numerator += dl(X, labels, distance, n_clusters)
#
#     N = elements
#     c_n_2 = (N * (N - 1)) / 2.0
#
#     cluster_sizes = count_cluster_sizes(labels, n_clusters)
#     nw = nW(n_clusters, cluster_sizes)
#
#     return numerator / (nw * (c_n_2 - nw))

# dl(..., distance) denotes the number of all object pairs in X,
# namely x_i and x_j, that fulfil two conditions:(a) x_i and x_j
# are in different clusters, and (b) de(x_i, x_j) < distance.
def dl(X, labels, distance):
    result = 0.0
    for i in range(0, len(labels) - 1):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                continue
            if euclidian_dist(X[i], X[j]) < distance:
                result += 1
    return result


# gamma, gamma index, mim is better
def gamma(X, labels, n_clusters):
    elements, ignore_columns = X.shape
    # dls & dists - matrix of size N x N
    dls = 0.0
    # dists = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
    #
    # for i in range(len(labels) - 1):
    #     for j in range(i + 1, len(labels)):
    #         dists[i][j] = euclidian_dist(X[i], X[j])

    for i in range(0, len(labels) - 1):
        for j in range(i + 1, len(labels)):
            if labels[i] != labels[j]:
                continue
            distance = euclidian_dist(X[i], X[j])
            dls += dl(X, labels, distance)

    # calculate denominator:
    N = elements
    c_n_2 = (N * (N - 1)) / 2.0
    nw = 0.0
    cluster_sizes = count_cluster_sizes(labels, n_clusters)
    for k in range(0, n_clusters):
        nw += cluster_sizes[k] * (cluster_sizes[k] - 1) / 2
    # TODO check if denominator is not 0.
    return dls / (nw * (c_n_2 - nw))


# cs, CS-index, min is better
def cs_index(X, labels, n_clusters):
    elements, ignore_columns = X.shape
    centroids = cluster_centroid(X, labels, n_clusters)
    cluster_sizes = count_cluster_sizes(labels, n_clusters)
    max_dists = [sys.float_info.min] * elements

    for i in range(0, elements):  # for every element
        for j in range(i, elements - 1):  # for every other
            if labels[i] != labels[j]: continue  # if they are in the same cluster
            # update the distance to the farthest element in the same cluster
            max_dists[i] = max(max_dists[i], euclidian_dist(X[i], X[j]))

    # max_dists contain for each element the farthest the his cluster

    numerator = 0.0
    for i in range(0, elements):
        if (cluster_sizes[labels[i]] != 0):
            numerator += max_dists[i] / cluster_sizes[labels[i]]

    denominator = 0.0
    for i in range(0, n_clusters - 1):
        min_centroids_dist = sys.float_info.max
        for j in range(i + 1, n_clusters):
            dist = euclidian_dist(centroids[i], centroids[j])
            min_centroids_dist = min(dist, min_centroids_dist)
        denominator += min_centroids_dist

    assert denominator != 0.0
    return numerator / denominator


# db_star, DB*-index, min is better
def db_star_index(X, labels, n_clusters):
    centroids = cluster_centroid(X, labels, n_clusters)
    cluster_sizes = count_cluster_sizes(labels, n_clusters)

    numerator = 0.0
    for k in range(0, n_clusters - 1):
        max_s_sum = sys.float_info.min
        min_centroids_dist = sys.float_info.max
        for l in range(k + 1, n_clusters):
            max_s_sum = max(max_s_sum,
                            s(X, k, cluster_sizes, labels, centroids)
                            + s(X, l, cluster_sizes, labels, centroids))
            min_centroids_dist = min(min_centroids_dist, euclidian_dist(centroids[k], centroids[l]))
        if min_centroids_dist == 0:
            min_centroids_dist = 0.00000000001
        numerator += max_s_sum / min_centroids_dist
    return numerator / n_clusters


# Score Function:
def bcd_score(X, labels, n_clusters, centroids, cluster_sizes):
    mean_x = np.mean(X, axis=0)
    numerator = 0.0
    for k in range(0, n_clusters):
        numerator += cluster_sizes[k] * euclidian_dist(centroids[k], mean_x)
    return (numerator / len(labels)) / n_clusters


def wcd_score(X, labels, n_clusters, centroids, cluster_sizes):
    numerator = 0.0
    for k in range(0, n_clusters):
        tmp = 0.0
        for i in range(0, len(labels)):
            if (labels[i] == k):
                tmp += euclidian_dist(X[i], centroids[k])
        numerator += tmp / cluster_sizes[k]
    return numerator


# sf, Score Function, max is better, added -
def sf(X, labels, n_clusters):
    centroids = cluster_centroid(X, labels, n_clusters)
    cluster_sizes = count_cluster_sizes(labels, n_clusters)

    bcd = bcd_score(X, labels, n_clusters, centroids, cluster_sizes)
    wcd = wcd_score(X, labels, n_clusters, centroids, cluster_sizes)

    distances = []

    for i in range(0, len(labels)):
        distances.append(euclidian_dist(X[i], centroids[labels[i]]))

    std = np.std(distances)

    try:
        t = bcd / std - wcd / std
        p = math.exp(t)
        p = math.exp(p)
    except OverflowError:
        p = sys.float_info.max
    return 1.0 / p  # - (1.0 - 1.0 / p) = 1/p - 1 (we do not need this -1). Avoid denormal numbers


# Sym Index:
def d_ps(X, labels, x_i, cluster_k_index, centroids):
    min1 = sys.float_info.max
    min2 = sys.float_info.max
    centroid = centroids[cluster_k_index]

    for j in range(0, len(labels)):
        if labels[j] != cluster_k_index:
            continue
        t = euclidian_dist(centroid + centroid - x_i, X[j])  # TODO: debug if addition is per coordinate
        if t < min1:
            min2 = min1
            min1 = t
        else:
            min2 = min(min2, t)

    return (min1 + min2) / 2.0


# sym, Sym Index, max is better, added -
def sym(X, labels, n_clusters):
    centroids = cluster_centroid(X, labels, n_clusters)

    numerator = sys.float_info.min
    for k in range(0, n_clusters - 1):
        for l in range(k, n_clusters):
            numerator = max(numerator, euclidian_dist(centroids[k], centroids[l]))

    denominator = 0.0
    for i in range(0, len(labels)):
        denominator += d_ps(X, labels, X[i], labels[i], centroids)
    return -((numerator / denominator) / n_clusters)


# cop, COP Index, min is better
def cop(X, labels, n_clusters):
    centroids = cluster_centroid(X, labels, n_clusters)
    numerators = [0.0] * n_clusters
    for i in range(0, len(labels)):
        numerators[labels[i]] += euclidian_dist(X[i], centroids[labels[i]])

    accumulator = 0.0
    for k in range(0, n_clusters):
        outer_min_dist = sys.float_info.max
        for i in range(0, len(labels) - 1):  # iterate elements outside cluster
            if labels[i] == k: continue
            inner_max_dist = sys.float_info.min
            for j in range(i, len(labels)):  # iterate inside cluster
                if labels[j] != k: continue
                inner_max_dist = max(inner_max_dist, euclidian_dist(X[i], X[j]))
            if inner_max_dist != sys.float_info.min:
                outer_min_dist = min(outer_min_dist, inner_max_dist)
        if (outer_min_dist != sys.float_info.max):
            accumulator += numerators[k] / outer_min_dist
    return accumulator / len(labels)


# sv, SV-Index, max is better, added -
def sv(X, labels, n_clusters):
    centroids = cluster_centroid(X, labels, n_clusters)
    cluster_sizes = count_cluster_sizes(labels, n_clusters)

    numerator = 0.0
    for k in range(0, n_clusters - 1):
        min_dist = sys.float_info.max
        for l in range(k + 1, n_clusters):
            min_dist = min(min_dist, euclidian_dist(centroids[k], centroids[l]))
        numerator += min_dist

    denominator = 0.0
    for k in range(0, n_clusters):
        list = []
        for i in range(0, len(labels)):
            if labels[i] != k:
                continue
            list.append(euclidian_dist(X[i], centroids[k]))

        # get sum of 0.1*|Ck| largest elements
        acc = 0.0
        max_n = heapq.nlargest(int(math.ceil(0.1 * cluster_sizes[k])), list)
        for i in range(0, len(max_n)):
            acc += max_n[i]
        denominator += acc * 10.0 / cluster_sizes[k]
    return - numerator / denominator


# OS-Index:
def a(X, labels, x_i, cluster_k_index, cluster_k_size):
    acc = 0.0
    for j in range(0, len(labels)):
        if labels[j] != cluster_k_index: continue
        acc += euclidian_dist(x_i, X[j])
    return acc / cluster_k_size


def b(X, labels, x_i, cluster_k_index, cluster_k_size):
    dists = []
    for j in range(0, len(labels)):
        if (labels[j] != cluster_k_index):
            dists.append(euclidian_dist(x_i, X[j]))

    # TODO: it can happen, that c_k_size if bigger than len(dists). Is it supposed to be so?

    acc = 0.0
    min_n = heapq.nsmallest(cluster_k_size, dists)
    for i in range(0, len(min_n)):
        acc += min_n[i]
    return acc / cluster_k_size


def ov(X, labels, x_i, cluster_k_index, cluster_k_size):
    a_s = a(X, labels, x_i, cluster_k_index, cluster_k_size)
    b_s = b(X, labels, x_i, cluster_k_index, cluster_k_size)

    if b_s == 0:
        b_s = 0.0000000000001
    if (b_s - a_s) / (b_s + a_s) < 0.4:
        return a_s / b_s
    else:
        return 0


# os, OS-Index, max is better, added -
def os(X, labels, n_clusters):
    centroids = cluster_centroid(X, labels, n_clusters)
    cluster_sizes = count_cluster_sizes(labels, n_clusters)

    numerator = 0.0
    for k in range(0, n_clusters):
        for i in range(0, len(labels)):
            if labels[i] != k: continue
            numerator += ov(X, labels, X[i], k, cluster_sizes[k])

    denominator = 0.0
    for k in range(0, n_clusters):
        l = []
        for i in range(0, len(labels)):
            if labels[i] != k:
                continue
            l.append(euclidian_dist(X[i], centroids[k]))

        # get sum of 0.1*|Ck| largest elements
        acc = 0.0
        max_n = heapq.nlargest(int(math.ceil(0.1 * cluster_sizes[k])), l)
        for i in range(0, len(max_n)):
            acc += max_n[i]

        denominator += acc * 10.0 / cluster_sizes[k]

    return - numerator / denominator


# SymDB:
def sym_s(X, labels, cluster_k_index, cluster_sizes, centroids):
    acc = 0.0
    for i in range(0, len(labels)):
        if labels[i] != cluster_k_index: continue
        acc += d_ps(X, labels, X[i], cluster_k_index, centroids)
    return acc / float(cluster_sizes[cluster_k_index])


# SymDB, Sym Davies-Bouldin index, min is better
def sym_db(X, labels, n_clusters):
    centroids = cluster_centroid(X, labels, n_clusters)
    db = 0
    cluster_sizes = count_cluster_sizes(labels, n_clusters)
    max_fraction = sys.float_info.min
    for k in range(0, n_clusters):
        for l in range(0, n_clusters):
            if k != l:
                fraction = (sym_s(X, labels, k, cluster_sizes, centroids) + sym_s(X, labels, l, labels, centroids)) \
                           / euclidian_dist(centroids[k], centroids[l])
                max_fraction = max(max_fraction, fraction)
        db += max_fraction
    db /= float(n_clusters)
    return db


# S_Dbw: (under construction)
# def euclidean_norm(x):
#     return np.linalg.norm(x)

def f(x_i, centroid_k, std):
    if std < euclidian_dist(x_i, centroid_k):
        return 0
    else:
        return 1


def mean(x_i, x_j):
    return (x_i + x_j) / 2


def den2(X, labels, centroids, k, l, std):
    acc = 0.0
    elements = len(X)
    for i in range(0, elements):
        if labels[i] == k or labels[i] == l:
            acc += f(X[i], mean(centroids[k], centroids[l]), std)
    return acc


def den1(X, labels, centroids, k, std):
    acc = 0.0
    elements = len(X)
    for i in range(0, elements):
        if labels[i] == k:
            acc += f(X[i], centroids[k], std)
    return acc


def normed_sigma(X):
    elements = len(X)
    sum = 0.0
    for i in range(0, elements):
        sum += X[i]
    avg = sum / elements
    sigma = 0.0

    for i in range(0, elements):
        sigma += (X[i] - avg) * (X[i] - avg)
    sigma = sigma / elements
    return math.sqrt(np.dot(sigma, np.transpose(sigma)))


def normed_cluster_sigma(X, labels, k):
    elements = len(X)
    sum = 0.0
    ck_size = 0
    for i in range(0, elements):
        if labels[i] == k:
            sum += X[i]
            ck_size += 1
    avg = sum / elements
    sigma = 0.0

    for i in range(0, elements):
        if labels[i] == k:
            sigma += (X[i] - avg) * (X[i] - avg)
    sigma = sigma / elements
    return math.sqrt(np.dot(sigma, np.transpose(sigma)))


def stdev(X, labels, n_clusters):
    sum = 0.0
    for k in range(0, n_clusters):
        sum += math.sqrt(normed_cluster_sigma(X, labels, k))
    sum /= n_clusters
    return sum


# s_dbw, S_Dbw index, min is better
def s_dbw(X, labels, n_clusters):
    centroids = cluster_centroid(X, labels, n_clusters)

    sigmas = 0.0
    for k in range(0, n_clusters):
        sigmas += normed_cluster_sigma(X, labels, k)
    sigmas /= n_clusters
    sigmas /= normed_sigma(X)

    stdev_val = stdev(X, labels, n_clusters)

    dens = 0.0
    for k in range(0, n_clusters-1):
        for l in range(k + 1, n_clusters):
            denominator = max(den1(X, labels, centroids, k, stdev_val), den1(X, labels, centroids, l, stdev_val))
            if denominator != 0:  # avoid zero division
                dens += den2(X, labels, centroids, k, l, stdev_val)


    dens /= n_clusters * (n_clusters - 1)
    return sigmas + dens
