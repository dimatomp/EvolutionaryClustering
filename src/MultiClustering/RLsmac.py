import time
from os import walk
from sys import argv

import numpy as np
import pandas as pd

from . import RLsmacAlgoEx as ae
from . import Constants
from .mab_solvers.MaxEiSolver import MaxEi
from .RLrfAlgoEx import RLrfrsAlgoEx
from .RLsmacEiAlgoEx import RLsmacEiAlgoEx
from .mab_solvers.Smx_R import SoftmaxR
from .mab_solvers.Smx_SRSU import SoftmaxSRSU
from .mab_solvers.UCB import UCB
from .mab_solvers.Softmax import Softmax
from .mab_solvers.UCB_SRSU import UCBsrsu
from .mab_solvers.UCB_SRU import UCBsru
from .mab_solvers.Uniform import Uniform

arglabel = None
if len(argv) == 7:
    script, argfile, argseed, argmetric, argiter, argbatch, argtl = argv
    algorithm = Constants.algorithm
elif len(argv) == 8:
    script, argfile, argseed, argmetric, argiter, argbatch, argtl, algorithm = argv
elif len(argv) == 9:
    script, argfile, argseed, argmetric, arglabel, argiter, argbatch, argtl, algorithm = argv
else:
    raise "Invalid error"


def configure_mab_solver(algorithm, metric, X, seed):
    """
    Creates and configures the corresponding MAB-solver.
    :param algorithm: algorithm to be used.
    """

    if algorithm.startswith("rl-ei"):
        algo_e = RLsmacEiAlgoEx(metric, X, seed, batch_size)
        mab_solver = Softmax(action=algo_e, tau=Constants.tau, is_fair=False, time_limit=time_limit)

        # Advanced MAB:
    elif algorithm.startswith("rfrsls-uni"):
        algo_e = RLrfrsAlgoEx(metric, X, seed, batch_size, expansion=100)
        mab_solver = Uniform(action=algo_e, time_limit=time_limit)
    elif algorithm.startswith("rfrsls-smx-R"):
        algo_e = RLrfrsAlgoEx(metric, X, seed, batch_size, expansion=100)
        mab_solver = SoftmaxR(action=algo_e, time_limit=time_limit)
    elif algorithm.startswith("rfrsls-ucb-SRU"):
        algo_e = RLrfrsAlgoEx(metric, X, seed, batch_size, expansion=100)
        mab_solver = UCBsru(action=algo_e, time_limit=time_limit)
    elif algorithm.startswith("rfrsls-ucb-SRSU"):
        algo_e = RLrfrsAlgoEx(metric, X, seed, batch_size, expansion=100)
        mab_solver = UCBsrsu(action=algo_e, time_limit=time_limit)
    elif algorithm.startswith("rfrsls-smx-SRSU"):
        algo_e = RLrfrsAlgoEx(metric, X, seed, batch_size, expansion=100)
        mab_solver = SoftmaxSRSU(action=algo_e, time_limit=time_limit)

        # Old MABs still supported
    else:
        algo_e = ae.RLsmacAlgoEx(metric, X, seed, batch_size)
        # choose algo:
        if algorithm == "rl-ucb-f-smac":
            mab_solver = UCB(action=algo_e, is_fair=True, time_limit=time_limit)
        elif algorithm == "rl-ucb1-smac":
            mab_solver = UCB(action=algo_e, is_fair=False, time_limit=time_limit)
        elif algorithm.startswith("rl-smx-smac"):
            mab_solver = Softmax(action=algo_e, tau=Constants.tau, is_fair=False, time_limit=time_limit)
        elif algorithm == "rl-smx-f-smac":
            mab_solver = Softmax(action=algo_e, tau=Constants.tau, is_fair=True, time_limit=time_limit)
        # elif argalgo == "rl-ucb1-rs":
        elif algorithm.startswith("rl-max-ei"):
            mab_solver = MaxEi(action=algo_e, optimizers=algo_e.smacs, time_limit=time_limit)
        else:
            raise "X3 algo: " + algorithm

    return mab_solver

def run(filename, seed, metric, output_file):
    data = pd.read_csv(Constants.experiment_path + filename)
    true_labels = None

    if arglabel:
        true_labels = np.array(data[arglabel])
        data = data.drop(arglabel, axis=1)

    X = np.array(data, dtype=np.double)
    name = str.split(filename, '.')[0]
    print(name)
    f = open(file=output_file, mode='a')

    # core part:
    mab_solver = configure_mab_solver(algorithm, metric, X, seed)

    start = time.time()
    # Random initialization:
    mab_solver.initialize(f, true_labels)
    time_init = time.time() - start

    start = time.time()
    # RUN actual Multi-Arm:
    its = mab_solver.iterate(iterations, f)
    time_iterations = time.time() - start

    print("#PROFILE: time spent in initialize: " + str(time_init))
    print("#PROFILE: time spent in iterations:" + str(time_iterations))

    algo_e = mab_solver.action

    f.write("Metric: " + metric + ' : ' + str(algo_e.best_val) + '\n')
    f.write("Algorithm: " + str(algo_e.best_algo) + '\n')
    f.write("# Target func calls: " + str(its * batch_size) + '\n')
    f.write("# Time init: " + str(time_init) + '\n')
    f.write("# Time spent: " + str(time_iterations) + '\n')
    f.write("# Arms played: " + str(mab_solver.n) + '\n')
    f.write("# Arms algos: " + str(Constants.algos) + '\n')

    try:
        f.write("# Arms avg time: " + str([np.average(plays) for plays in mab_solver.spendings]) + '\n')
    except:
        f.write("# Arms avg time: []")
        pass

    f.write(str(algo_e.best_param) + "\n\n")

    f.write("SMACS: \n")
    if hasattr(algo_e, "smacs"):
        for s in algo_e.smacs:
            try:
                stats = s.get_tae_runner().stats
                t_out = stats._logger.info
                stats._logger.info = lambda x: f.write(x + "\n")
                stats.print_stats()
                stats._logger.info = t_out
            except:
                pass

        f.write("\n")
        for i in range(0, Constants.num_algos):
            s = algo_e.smacs[i]
            _, Y = s.solver.rh2EPM.transform(s.solver.runhistory)
            f.write(Constants.algos[i] + ":\n")
            f.write("Ys:\n")
            for x in Y:
                f.write(str(x[0]))
                f.write("\n")
            f.write("-----\n")

    f.write("###\n")
    f.write("\n\n")

    if algorithm.startswith("rl-max-ei"):
        log = mab_solver.tops_log
    elif algorithm.startswith("rl-ei"):
        log = algo_e.tops_log
    else:
        log = []

    for i in range(0, len(log)):
        f.write(str(i + 1) + ": " + str(log[i]))
        f.write("\n")

    f.flush()


def run_all(filename):
    name = str.split(filename, '.')[0]

    for seed in seeds:
        ans = 'result/' + algorithm + '_' + name + '_' + str(seed) + '_i' + str(iterations) \
              + '_b' + str(batch_size) + '_t' + str(argtl)
        for metric in metrics:
            ans += "," + metric

        ans += ".salt=" + str(int(np.random.uniform(0, 1000000000)))
        ans += '.txt'

        for metric in metrics:
            run(filename, seed, metric, ans)


# main
print("RL-SMAC search for " + str(argfile) + " " + str(argseed) + " "
      + str(argmetric) + " " + str(argiter) + " " + str(argbatch))

if (argseed is None) or (argseed == "all"):
    seeds = Constants.seeds
else:
    seeds = [int(argseed)]

if (argmetric is None) or (argmetric == "all"):
    metrics = Constants.metrics
else:
    # short way:
    if argmetric == "ch":
        metrics = [Constants.cal_har_metric]
    elif argmetric == "sil":
        metrics = [Constants.silhouette_metric]
    else:
        metrics = [argmetric]

if (argiter is None) or (argiter == ""):
    iterations = Constants.bandit_iterations
else:
    iterations = int(argiter)

if (argbatch is None) or (argbatch == ""):
    batch_size = Constants.batch_size
else:
    batch_size = int(argbatch)

if argtl is None:
    time_limit = Constants.tuner_timeout
else:
    time_limit = int(argtl)

if argfile == "all":
    for (dirpath, dirnames, files) in walk(Constants.experiment_path):
        for ii in range(0, len(files)):
            file = files[ii]
            run_all(file)
else:
    run_all(argfile)
