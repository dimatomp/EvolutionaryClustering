import math
import sys
import time
import traceback

import numpy as np
import pandas as pd
from . import Metric
from . import Constants
from sys import argv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

script, argmetrics, repeat = argv


def plot(data, labels, title):
    plt.figure(figsize=(9, 5))
    plt.boxplot(data, labels=labels)

    plt.xlabel("Количество объектов")
    plt.ylabel("Время, секунды")
    plt.title(title)
    plt.tight_layout()

    plt.savefig('./pics/fig_' + title)
    plt.clf()


if argmetrics == "ch":
    argmetrics = Constants.cal_har_metric
elif argmetrics == "sil":
    argmetrics = Constants.silhouette_metric

data = pd.read_csv(Constants.experiment_path + "krvskp.csv", header=None)
all_data = np.array(data, dtype=np.double)
np.random.shuffle(all_data)

width = 10

# for metrics in Constants.metrics:
metrics = argmetrics
# metrics = Constants.silhouette_metric

tss = []
tss_labels = []

total_start = time.time()

f = open(file="./pics/" + metrics + "_100-2000" + ".txt", mode='w')
for size in range(100, 2100, 100):
    n = int(size)
    data_set = all_data[0:n, 0:int(width)]

    print("# " + metrics + ": krvskp: " + str(data_set.shape))

    try:
        ts = []
        rs = []
        for n_clusters in range(2, 15):
            labels = np.random.randint(0, n_clusters, n)

            # make sure all cluster are present:
            for c in range(0, n_clusters):
                labels[c] = c
            np.random.shuffle(labels)

            # print(labels)
            for i in range(0, int(repeat)):
                np.random.shuffle(labels)
                start = time.time()
                res = Metric.metric(data_set, n_clusters, labels, metrics)
                spent = time.time() - start
                ts.append(spent)
                rs.append(res)
                if math.fabs(res) > 1000 or spent > 100:
                    f.write("---> t=" + str(spent) + "s, res=" + res + ", l:" + str(labels.tolist()))

        # print("ts= " + str(ts))
        # print("rs= " + str(rs))
        # ts = np.array(ts)

        mean = np.mean(ts)
        std = np.std(ts)
        mx = np.max(ts)
        mn = np.min(ts)

        s = "Stats for " + "krvskp:" + str(data_set.shape) + " on " + metrics + "\n" \
            + "\t # " + str(len(ts)) + "\n" \
            + "\t mean " + str(mean) + "\n" \
            + "\t std " + str(std) + "\n" \
            + "\t max " + str(mx) + "\n" \
            + "\t min " + str(mn) + "\n"
        print(s)

        f.write(s)
        f.write("\n")
        f.write("ts: " + str(ts))
        f.write("\n")
        f.write("rs: " + str(rs))
        f.write("\n")
        f.flush()

        tss.append(ts)
        tss_labels.append(size)
    except:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        del exc_info

plot(tss, tss_labels, metrics + " ; ( x , " + str(10) + " )")

print("TOTAL: " + str(time.time() - total_start))
