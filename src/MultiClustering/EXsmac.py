import time
from os import walk
from sys import argv

import numpy as np
import pandas as pd
import sys

from . import EXsmacAlgoExecutor as ae
from . import Constants
from .ExThreads import LimitType

script, argfile, argseed, argmetric, argbudget = argv


def run(filename, seed, metric, output_file):
    data = pd.read_csv(Constants.experiment_path + filename)
    X = np.array(data, dtype=np.double)
    name = str.split(filename, '.')[0]
    print(name)
    f = open(file=output_file, mode='a')

    print("\nStart metric " + metric)
    start = time.time()

    # core part:
    # use LimitType.TIME or LimitType.CALLS to limit smac by *budget* seconds or calls respectively
    algo_e = ae.EXsmacAlgoExecutor(Constants.num_algos, metric, X, seed, LimitType.TIME)
    algo_e.apply(budget)

    time_total = time.time() - start
    print("#PROFILE: time spent:" + str(time_total))

    f.write("Metric: " + metric + ' : ' + str(algo_e.best_val) + '\n')
    f.write("Algorithm: " + str(algo_e.best_algo) + '\n')
    f.write("# Limit Type: " + str(LimitType.TIME) + '\n')
    f.write("# Budget: " + str(budget) + '\n')
    f.write("# Time spent: " + str(time_total) + '\n')
    f.write(str(algo_e.best_param) + "\n\n")
    f.write(str(algo_e.per_algo) + '\n')
    f.write("\n")

    f.write("SMACS: \n")
    for s in algo_e.smacs:
        try:
            stats = s.get_tae_runner().stats
            t_out = stats._logger.info
            stats._logger.info = lambda x: f.write(x + "\n")
            stats.print_stats()
            stats._logger.info = t_out
            f.write(str(s.get_runhistory()) + "\n")
        except:
            pass

    f.write("###\n")
    f.flush()


def run_all(filename):
    name = str.split(filename, '.')[0]

    for seed in seeds:
        ans = 'result/ex-smac_' + name + '_' + str(seed) + '_i0' + '_t' + str(budget)
        for metric in metrics:
            ans += "," + metric

        ans += ".salt=" + str(int(np.random.uniform(0,1000000000)))
        ans += '.txt'

        for metric in metrics:
            run(filename, seed, metric, ans)


# main
print("RL-SMAC search for " + str(argfile) + " " + str(argseed) + " "
      + str(argmetric) + " budget:" + str(argbudget))

if (argseed is None) or (argseed == "all"):
    seeds = Constants.seeds
else:
    seeds = [int(argseed)]

if (argmetric is None) or (argmetric == "all"):
    metrics = Constants.metrics
else:
    if argmetric == "ch":
        metrics = [Constants.cal_har_metric]
    elif argmetric == "sil":
        metrics = [Constants.silhouette_metric]
    else:
        metrics = [argmetric]

if (argbudget is None) or (argbudget == ""):
    budget = Constants.bandit_iterations * Constants.batch_size
else:
    budget = int(argbudget)

if argfile == "all":
    for (dirpath, dirnames, files) in walk(Constants.experiment_path):
        for ii in range(0, len(files)):
            file = files[ii]
            run_all(file)
else:
    run_all(argfile)
