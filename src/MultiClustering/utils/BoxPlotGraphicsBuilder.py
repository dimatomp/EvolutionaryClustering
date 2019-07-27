import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import markers
from matplotlib.colors import colorConverter as cc

mab_solvers = ["ex-smac", "rl-smac"]
# times = ["1200", "600", "400", "300", "200"]
times = ["600", "400", "300", "200", "100", "50", "25"]
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


def plotoOverlayBoxes(data, labels, title):
    plt.figure(figsize=(9, 5))

    for d in data:
        plt.boxplot(x=d, labels=labels, )

    plt.xlabel("Бюджет")
    plt.ylabel("Мера")
    plt.title(title)
    plt.tight_layout()

    plt.savefig('./pics/fig_' + title)
    plt.clf()
    plt.close()


def plotBoxes(data, labels, title):
    plt.figure(figsize=(9, 5))
    plt.boxplot(x=data, labels=labels)

    plt.xlabel("Бюджет")
    plt.ylabel("Мера")
    plt.title(title)
    plt.tight_layout()

    plt.savefig('./pics/fig_' + title)
    plt.clf()
    plt.close()



def draw_graphics(stats):
    for m in metrics:
        for d in ds_by_size:

            # lines = {a: {"l": [], "m": [], "u": []} for a in mab_solvers}

            groups = {a: {t: [] for t in times} for a in mab_solvers}

            for t in times:
                for a in mab_solvers:
                    if t in stats[a][m][d]:
                        runs = stats[a][m][d][t]
                        vs = [r.value for r in runs]
                        nevs = list(filter(lambda x: x != "", vs))
                        fvs = [float(f) for f in nevs]
                        # filter bad results and crashes:
                        fvs = list(filter(lambda x: np.math.fabs(x) <= 10000000, fvs))

                        if len(fvs) != 0:
                            # # choose results
                            # fvs = np.sort(fvs)
                            # if a == mab_solvers[0]:
                            #     fvs = fvs[-10:]
                            # else:
                            #     fvs = fvs[0:10]
                            groups[a][t] = fvs

            for a in mab_solvers:
                to_plot = []
                for t in times:
                    to_plot.append(groups[a][t])
                plotBoxes(to_plot, times, a + ", " + d + ", " + m)


