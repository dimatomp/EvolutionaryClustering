import traceback

import pandas as pd
import numpy as np
import os
import ast
from os import walk
from sys import argv
from sys import exc_info

import sys
from .. import Constants
from . import BoxPlotGraphicsBuilder
from .ShadeGraphicsBuilder import draw_graphics

print(argv)
script, path_in, f_out = argv

algos = ["ex-smac", "rl-smac", "rl-smx-smac",
         "rl-smx-smac-20", "rl-smx-smac-100", "rl-smx-smac-500",
         "rl-ucb1-smac", "rl-ucb-fair-smac",
         "ex-rs", "rl-ei-new-old", "rl-ei-old-new", "rl-max-ei"]
stats = {}

for a in algos:
    stats[a] = {}


class Run:
    def __init__(self, metric, dataset, time_limit, value, time_spent,
                 per_algo, tae_runs, plays, iterations=None):
        self.metric = metric
        self.dataset = dataset
        self.time_limit = time_limit
        self.value = value
        self.time_spent = time_spent
        self.per_algo = per_algo
        self.iterations = iterations
        self.arm_plays = plays
        self.tae_runs = tae_runs


def skip():
    f.write("\t")
    st.write("\t")
    fr.write("\t")
    fc.write("\t")


def new_line():
    f.write("\n")
    st.write("\n")
    fr.write("\n")
    fc.write("\n")


arm_plays = {}

for root, subdirs, files in walk(path_in):
    # print(str(files))
    # print(str(subdirs))
    # print(str(root))
    for filename in files:
        file_path = os.path.join(root, filename)
        # for ii in range(0, len(files)):
        #     filename = files[ii]
        algo = filename.split("_")[0]
        dataset = filename.split("_")[1]
        metric = filename.split(",")[1].split(".")[0]

        if algo == "ex-smac" or algo == "ex-rs":
            budget = filename.split(",")[0].split("_")[4][1:]  # b600 or t600 etc
        else:
            t = filename.split(",")[0].split("_")
            # budget = str(600) if len(t) == 5 else t[5][1:]
            try:
                budget = t[5][1:]
            except:
                print("ERROR: " + root + "/" + filename, file=sys.stderr)

        print("Process: " + algo + " - " + dataset + " - " + metric + " - t" + budget)

        if not metric in stats[algo]:
            stats[algo][metric] = {}
        if not dataset in stats[algo][metric]:
            stats[algo][metric][dataset] = {}

        if not metric in arm_plays:
            arm_plays[metric] = {}

        with open(os.path.join(root, filename)) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]

        if not dataset in arm_plays[metric]:
            arm_plays[metric][dataset] = {}
            arm_plays[metric][dataset]["p"] = {}
            arm_plays[metric][dataset]["t"] = []

        if not budget in arm_plays[metric][dataset]["p"]:
            arm_plays[metric][dataset]["p"][budget] = []

        value = ""
        time_spent = ""
        per_algo = ""
        tae_runs = []
        plays = ""
        for line in lines:
            if line.startswith("Metric:"):
                value = line.split(": ")[2]
                continue
            if line.startswith("# Time spent: "):
                time_spent = line.split(": ")[1]
            if line.startswith("{"):
                per_algo = line
            if line.startswith("#Target algorithm runs: "):
                tae_runs.append(int(line.split(": ")[1].split(" /")[0]))
            if line.startswith("# Target func calls: "):
                plays = line.split(": ")[1]
            if line.startswith("# Arms played: "):
                arm_plays[metric][dataset]["p"][budget].append(line.split(": ")[1])
            if line.startswith("# Arms avg time: "):
                arm_plays[metric][dataset]["t"].append(line.split(": ")[1])

        if not budget in stats[algo][metric][dataset]:
            stats[algo][metric][dataset][budget] = []

        stats[algo][metric][dataset][budget].append(
            Run(metric, dataset, budget, value, time_spent, per_algo, tae_runs, plays))

# BoxPlotGraphicsBuilder.draw_graphics(stats)
# draw_graphics(stats)
#
# exit(0)

algos = ["ex-smac", "rl-smac", "rl-max-ei"]
times = ["1200", "600", "400", "300", "200", "100", "50", "25"]
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

f = open(file=f_out, mode='w')
fr = open(file=f_out + ".runs.txt", mode='w')
fc = open(file=f_out + ".calls.txt", mode='w')
st = open(file=f_out + ".stddev.txt", mode='w')

for t in times:
    for a in algos:
        f.write(t + "\t")
f.write("\n")

for t in times:
    for a in algos:
        f.write(a + "\t")
f.write("\n")

for m in metrics:
    for d in ds_by_size:
        for t in times:
            for a in algos:
                if d not in stats[a][m]:
                    skip()
                    continue

                if t in stats[a][m][d]:
                    runs = stats[a][m][d][t]
                    vs = [r.value for r in runs]
                    nevs = list(filter(lambda x: x != "", vs))
                    fvs = [float(f) for f in nevs]
                    # filter bad results and crashes:
                    fvs = list(filter(lambda x: np.math.fabs(x) <= 10000000, fvs))

                    if a == algos[0]:
                        tae_runs = [r.tae_runs for r in runs]
                        tae_runs = list(filter(lambda l: len(l) == 7, tae_runs))  # list of 7-element lists
                        sums_tae_runs = [np.sum(tr) for tr in tae_runs]
                    else:
                        plays = [r.arm_plays for r in runs]
                        plays = list(filter(lambda x: x != "", plays))
                        sums_tae_runs = [int(x) for x in plays]

                    if len(fvs) == 0:
                        skip()
                    else:
                        # try:
                        # choose results
                        # fvs = np.sort(fvs)
                        # if a == algos[0]:
                        #     fvs = fvs[-10:]
                        # else:
                        #     fvs = fvs[0:10]
                        
                        f.write(str(np.average(fvs)) + "\t")
                        st.write(str(np.std(fvs)) + "\t")
                        fr.write(str(len(fvs)) + "\t")
                        if len(sums_tae_runs) != 0:
                            fc.write(str(np.average(sums_tae_runs)) + "\t")
                        else:
                            fc.write("\t")
                        # except:
                        #     traceback.print_exception(*exc_info())

                else:
                    skip()

        new_line()

f.flush()
f.close()
st.flush()
st.close()
fr.flush()
fr.close()
fc.flush()
fc.close()

# f = sys.stdout # open(file=sys.stdout, mode='w')
# for m in metrics:
#     for d in ds_by_size:
#         tn = 0
#         avg_times = np.array([0.0] * 7)
#         for t in arm_plays[m][d]["t"]:
#             try:
#                 tl = ast.literal_eval(t)
#                 if len(tl) != 7:
#                     continue
#                 avg_times = avg_times + np.array(tl)
#                 tn += 1
#             except:
#                 pass
#
#         avg_times /= tn
#         # print(avg_plays)
#         for t in avg_times:
#             f.write(str(t) + "\t")
#         f.write("\n")
#
#         for t in times:
#             pn = 0
#             avg_plays = np.array([0.0] * 7)
#             for p in arm_plays[m][d]["p"][t]:
#                 p = p.replace(" ", ",")
#                 p = p.replace(",,", ",")
#                 p = p.replace("[,", "[")
#                 p = p.replace(",]", "]")
#                 try:
#                     pl = ast.literal_eval(p)
#                     if len(pl) != 7:
#                         continue
#                     avg_plays = avg_plays + np.array(pl)
#                     pn += 1
#                 except:
#                     pass
#             avg_plays /= pn
