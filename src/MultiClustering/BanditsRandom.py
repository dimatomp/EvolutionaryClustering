import time
from os import walk
from sys import argv

import numpy as np
import pandas as pd

from . import AlgorithmExecRand as ae
from . import Constants
from . import Metric
from .mab_solvers import Softmax as sf

script, argfile, argseed = argv


def run(filename):
    run_start = time.time()
    Metric.global_trace = {}  # clear all trace from previous dataset
    data = pd.read_csv(Constants.experiment_path + filename)
    mas = str.split(filename, '_')
    name = mas[0]
    if (len(mas) < 2):
        name = name[:-4]
    print(name)
    X = np.array(data, dtype=np.double)


    metrics = [
        # Constants.davies_bouldin_metric, # 1
        Constants.dunn_metric, # 2
        Constants.cal_har_metric,  # 3, from scikit-learn
        Constants.silhouette_metric, # 4, from scikit-learn
        Constants.dunn31_metric,  # 5
        Constants.dunn41_metric, # 6
        Constants.dunn51_metric,  # 7
        Constants.dunn33_metric,  # 8
        Constants.dunn43_metric, # 9
        Constants.dunn53_metric,  # 10
        # Constants.gamma_metric,  # 11  # BROKEN
        # Constants.cs_metric, # 12
        # Constants.db_star_metric, # 13
        # Constants.sf_metric, # 14
        # Constants.sym_metric, # 15
        # Constants.cop_metric,  # 16
        # Constants.sv_metric, # 17
        # Constants.os_metric,  # 18
        # Constants.s_dbw_metric, # 19  # BROKEN
        # Constants.c_ind_metric # 20
    ]

    if (argseed is None) or (argseed == "all"):
        seeds = Constants.seeds
    else:
        seeds = [int(argseed)]

    for seed in seeds:
        ans = 'result/bandit_' + name + '_' + str(seed) + '_'
        for metric in metrics:
            ans += "," + metric
        ans += '.txt'
        f = open(ans, 'w', 1)
        for metric in metrics:
            print("\nStart metric " + metric)

            algo_e = ae.AlgorithmExecutor(Constants.num_algos, metric, X, seed)
            soft_max = sf.Softmax(algo_e, Constants.num_algos, Constants.tau)

            start = time.time()
            soft_max.initialize(f)
            print("#PROFILE: time spent in initialize()" + str(time.time() - start))

            soft_max.iterate(Constants.bandit_iterations, f)

            f.write("Metric: " + metric + ' : ' + str(algo_e.best_val) + '\n')
            f.write("Algorithm: " + str(algo_e.best_algo) + '\n')
            f.write(str(algo_e.best_param) + '\n\n')

        f.close()

        print("#PROFILE: TOTAL time consumed by run: " + str(time.time() - run_start))

        if len(Metric.global_trace) != 0:
            s = 0.0
            for i in range(0, len(Metric.global_trace[metric])):
                s = s + Metric.global_trace[metric][i]
            print("#PROFILE: time spent in calculating metrics (" + str(len(Metric.global_trace[metric]))
                  + " calls) " + str(metric) + ": " + str(s))

            print("#PROFILE: average metrics call consumes " + str(s / len(Metric.global_trace[metric])))
            print("Metrics " + str(metric) + " calls: " + Metric.global_trace[metric])
        else:
            print("#PROFILE: no metrics calculation outside SMAC runs found")



if argfile == "all":
    for (dirpath, dirnames, files) in walk(Constants.experiment_path):
        for ii in range(0, len(files)):
            file = files[ii]
            run(file)
else:
    run(argfile)
