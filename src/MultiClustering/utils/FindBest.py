import traceback

import pandas as pd
import numpy as np
import os
import ast
from os import walk
from sys import argv
import matplotlib.pyplot as plt

import sys

from sklearn.decomposition import PCA

from .. import Constants
from ..RLthreadBase import ClusteringArmThread

script, path_in, f_out = argv

algos = ["ex-smac", "rl-smac", "rl-smx-smac",
         "rl-smx-smac-20", "rl-smx-smac-100", "rl-smx-smac-500",
         "rl-ucb1-smac", "rl-ucb-fair-smac",
         "ex-rs", "rl-ei-new-old", "rl-ei-old-new", "rl-max-ei"]
stats = {}
# stats -> dataset -> metric -> algo -> (best_ val, config)

def skip():
    f.write("\t")


def new_line():
    f.write("\n")


def draw(x, labels, dataset, algo, metrics, mn=None):
    X_res = PCA(n_components=2).fit_transform(x)
    plt.figure(figsize=(9, 6))

    plt.scatter(X_res[:, 0], X_res[:, 1], c=labels)
    plt.title(algo + "(" + dataset + "), " + metrics + ". %s кластеров" % len(np.unique(labels)))


    # val = ""
    # if mn is not None:
    #     val = "val=" + str("{:10.4f}".format(mn))

    plt.savefig('./def_pics/fig_' + dataset + "_" + algo + "_" + metrics)
    plt.clf()
    plt.close()


for root, subdirs, files in walk(path_in):
    for filename in files:
        file_path = os.path.join(root, filename)

        algo = filename.split("_")[0]
        dataset = filename.split("_")[1]
        metric = filename.split(",")[1].split(".")[0]

        if algo == "ex-smac" or algo == "ex-rs":
            budget = filename.split(",")[0].split("_")[4][1:]  # b600 or t600 etc
        else:
            t = filename.split(",")[0].split("_")
            try:
                budget = t[5][1:]
            except:
                print("ERROR: " + root + "/" + filename, file=sys.stderr)

        print("Process: " + algo + " - " + dataset + " - " + metric + " - t" + budget)

        if not dataset in stats:
            stats[dataset] = {}
        if not metric in stats[dataset]:
            stats[dataset][metric] = {}
        # if not algo in stats[dataset][metric]:
        #     stats[dataset][metric][algo] = {}

        with open(os.path.join(root, filename)) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]

        value = sys.float_info.max
        winner = ""
        params = []
        is_conf = False

        for line in lines:
            if line.startswith("Metric:"):
                value = float(line.split(": ")[2])
                continue
            elif line.startswith("Algorithm:"):
                winner = line.split(": ")[1]
            elif line.startswith("Configuration:"):
                is_conf = True
            elif line == "":
                is_conf = False
            elif is_conf:
                if not line.startswith("verbose"):
                    rep = line.replace(", Value", "")
                    rep = rep.replace(":", "':")
                    params.append("'" + rep)

        if winner not in stats[dataset][metric]:
            stats[dataset][metric][winner] = (value, params)
        elif stats[dataset][metric][winner][0] <= value:
            stats[dataset][metric][winner] = (value, params)

# BoxPlotGraphicsBuilder.draw_graphics(stats)
# draw_graphics(stats)
# exit(0)

metrics = ["calinski-harabasz", "silhouette", "cop"]
ds_by_size = [
    "iris",
    "glass",
    # "haberman",
    "wholesale",
    "indiandiabests",
    "yeast",
    "krvskp"
]
c_algos = ['KMeans', 'Affinity_Propagation', 'Mean_Shift', 'Ward	DBSCAN', 'Gaussian_Mixture',
           'Bayesian_Gaussian_Mixture']

f = open(file=f_out, mode='w')

for m in metrics:
    for d in ds_by_size:
        mn = sys.float_info.max
        best_c = ""
        params = ""

        for c in c_algos:
            if c not in stats[d][m]:
                continue

            if stats[d][m][c][0] <= mn:
                mn = stats[d][m][c][0]
                best_c = c
                params = stats[d][m][c][1]

        # f.write(str(mn) + "\t" + best_c + "\t" + ", ".join(params) + "\n")

        for c in c_algos:
            if c not in stats[d][m]:
                continue

            if stats[d][m][c][0] == mn:
                try:
                    cfg = ast.literal_eval("{" + ", ".join(stats[d][m][c][1]) + "}")

                    data = pd.read_csv("datasets/unified/" + d + ".csv")
                    X = np.array(data, dtype=np.double)

                    cl = ClusteringArmThread(c, m, X, None)
                    labels = cl.cluster(cfg)

                    draw(X, labels, d, c, m, mn)

                except:
                    print("fail: " + d + ", " + m, file=sys.stderr)
                    print(stats[d][m], file=sys.stderr)


f.flush()
f.close()
