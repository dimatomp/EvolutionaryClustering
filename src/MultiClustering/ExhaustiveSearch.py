import time
from os import walk
from sys import argv

import numpy as np
import pandas as pd

from . import Constants
from . import Metric
from . import ExThreads as t

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

    algos = {Constants.dbscan_algo: 0,
             Constants.kmeans_algo: 0,
             Constants.affinity_algo: 0,
             Constants.mean_shift_algo: 0,
             Constants.ward_algo: 0,
             Constants.gm_algo: 0,
             Constants.bgm_algo: 0}
    metrics = [
        Constants.davies_bouldin_metric,  # 1
        Constants.dunn_metric,  # 2
        Constants.cal_har_metric,  # 3, from scikit-learn
        Constants.silhouette_metric,  # 4, from scikit-learn
        Constants.dunn31_metric,  # 5
        Constants.dunn41_metric,  # 6
        Constants.dunn51_metric,  # 7
        Constants.dunn33_metric,  # 8
        Constants.dunn43_metric,  # 9
        Constants.dunn53_metric,  # 10
        # #onstants.gamma_metric,  # 11  # broken
        Constants.cs_metric,  # 12
        Constants.db_star_metric,  # 13
        Constants.sf_metric,  # 14
        Constants.sym_metric,  # 15
        Constants.cop_metric,  # 16
        Constants.sv_metric,  # 17
        Constants.os_metric,  # 18
        Constants.s_dbw_metric,  # 19
        Constants.c_ind_metric  # 20
    ]

    saved_parameters = dict()
    for metric in metrics:
        for algo in algos:
            saved_parameters[metric] = dict()
            saved_parameters[metric][algo] = ""

    if (argseed is None) or (argseed == "all"):
        seeds = Constants.seeds
    else:
        seeds = [int(argseed)]

    for seed in seeds:
        ans = 'result/exhaustive_' + name + '_' + str(seed) + '_'
        for metric in metrics:
            ans += "," + metric
        ans += '.txt'
        f = open(ans, 'w', 1)

        best_metrics = dict()

        for metric in metrics:
            best_algo = "-1"
            best_params = dict()
            best_val = Constants.best_init

            print('Start meteric ' + metric)
            th = [0] * 7
            th[0] = t.km_thread(metric, X, seed)
            # th[0].start()
            th[1] = t.aff_thread(metric, X, seed)
            # th[1].start()
            th[2] = t.ms_thread(metric, X, seed)
            # th[2].start()
            th[3] = t.w_thread(metric, X, seed)
            # th[3].start()
            th[4] = t.db_thread(metric, X, seed)
            # th[4].start()
            th[5] = t.gm_thread(metric, X, seed)
            # th[5].start()
            th[6] = t.bgm_thread(metric, X, seed)
            # th[6].start()

            time_per_algo = []
            for i in range(0, Constants.num_algos):
                start = time.time()

                th[i].start()
                th[i].join()

                time_per_algo.append(time.time() - start)

                print(('              For algo ' + th[i].thread_name + ' with metric ' + th[i].metric +
                       ' lowest function value found: %f' % th[i].value))
                print(('              Parameter setting %s' % th[i].parameters))
                if th[i].value < best_val:
                    best_val = th[i].value
                    best_algo = th[i].thread_name
                    best_params = th[i].parameters
                saved_parameters[metric][th[i].thread_name] = th[i].parameters

            if best_algo != "-1":
                algos[best_algo] += 1
                if (best_algo not in best_metrics):
                    best_metrics[best_algo] = [metric]
                else:
                    best_metrics[best_algo].append(metric)

            f.write("Metric: " + metric + ' : ' + str(best_val) + '\n')
            f.write("Algorithm: " + str(best_algo) + '\n')
            f.write(str(best_params) + '\n\n')

            s = 0.0
            for i in range(0, len(time_per_algo)):
                s = s + time_per_algo[i]
            print("#PROFILE: sum of time consumed by iterations: " + str(s))
            print("#PROFILE: time_per_algo: \n" + str(time_per_algo))

        f.close()

        print("#PROFILE: TOTAL time consumed by run: " + str(time.time() - run_start))

        if len(Metric.global_trace) != 0:
            s = 0.0
            for i in range(0, len(Metric.global_trace[metric])):
                s = s + Metric.global_trace[metric][i]
            print("#PROFILE: time spent in calculating metrics (" + str(len(Metric.global_trace[metric]))
                  + " calls) " + str(metric) + ": " + str(s))
            print("#PROFILE: average metrics call consumes " + str(s / len(Metric.global_trace[metric])))
            print(Metric.global_trace[metric])
        else:
            print("#PROFILE: no metrics calculation outside SMAC runs found")
        print(saved_parameters)


if argfile == "all":
    for (dirpath, dirnames, files) in walk(Constants.experiment_path):
        for ii in range(0, len(files)):
            file = files[ii]
            run(file)
else:
    run(argfile)
