import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import markers
from matplotlib.colors import colorConverter as cc

algos = ["ex-smac", "rl-smac", "rl-ucb1-smac"]
# times = ["1200", "600", "400", "300", "200"]
times = ["1200", "600", "400", "300", "200", "100", "50", "25"]
metrics = ["calinski-harabasz", "silhouette", "cop"]

ds_by_size = [
    "iris",
    "glass",
    # "haberman",
    "wholesale",
    "indiandiabests",
    "yeast"
    ,
    # "krvskp"
]


def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None, mx=None, mn=None):
    # plot the shaded range of the confidence intervals
    x = [float(t) for t in times]
    plt.fill_between(x=x, y1=ub, y2=lb, color=color_shading, alpha=.3)
    # plot the mean on top
    print("draw: " + str(mean) + " of " + str(color_mean))
    plt.plot(x, mean, color_mean)
    if mx is not None:
        plt.scatter(x=x, y=mx, c=color_mean, s=10, marker="^")
        plt.scatter(x=x, y=mn, c=color_mean, s=10, marker="v")


class LegendObject(object):
    def __init__(self, facecolor='red', edgecolor='white', dashed=False):
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.dashed = dashed

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle(
            # create a rectangle that is filled with color
            [x0, y0], width, height, facecolor=self.facecolor,
            # and whose edges are the faded color
            edgecolor=self.edgecolor, lw=3)
        handlebox.add_artist(patch)

        # if we're creating the legend for a dashed line,
        # manually add the dash in to our rectangle
        if self.dashed:
            patch1 = mpatches.Rectangle(
                [x0 + 2 * width / 5, y0], width / 5, height, facecolor=self.edgecolor,
                transform=handlebox.get_transform())
            handlebox.add_artist(patch1)

        return patch


def draw_graphics(stats):
    for m in metrics:
        for d in ds_by_size:

            lines = {a: {"l": [], "m": [], "u": [], "mx": [], "mn": []} for a in algos}

            for t in times:
                for a in algos:
                    if t in times:

                        # TODO temporal fix for 1200. experiments will be soon
                        if t in stats[a][m][d]:
                            runs = stats[a][m][d][t]
                        elif t.startswith("1200"):
                            runs = stats[a][m][d]["600"]
                        else:
                            continue

                        vs = [r.value for r in runs]
                        nevs = list(filter(lambda x: x != "", vs))
                        fvs = [float(f) for f in nevs]
                        # filter bad results and crashes:
                        fvs = list(filter(lambda x: np.math.fabs(x) <= 10000000, fvs))

                        if len(fvs) != 0:
                            # choose results
                            fvs = np.sort(fvs)
                            if a == algos[0]:
                                fvs = fvs[-10:]
                            else:
                                fvs = fvs[0:10]

                            mean = np.average(fvs)
                            std = np.std(fvs)
                            lines[a]["l"].append(mean - std * 3)
                            lines[a]["m"].append(mean)
                            lines[a]["u"].append(mean + std * 3)
                            lines[a]["mx"].append(np.max(fvs))
                            lines[a]["mn"].append(np.min(fvs))

            plt.figure(1, figsize=(7, 2.5))
            colors = ['black', 'blue', 'green']
            for i in range(0, len(algos)):
                a = algos[i]
                plot_mean_and_CI(np.array(lines[a]["m"]), np.array(lines[a]["u"]), np.array(lines[a]["l"]),
                                 color_mean=colors[i], color_shading=colors[i],
                                 mx=np.array(lines[a]["mx"]), mn=np.array(lines[a]["mn"])
                                 )

            bg = np.array([1, 1, 1])  # background of the legend is white
            # with alpha = .5, the faded color is the average of the background and color
            colors_faded = [(np.array(cc.to_rgb(color)) + bg) / 2.0 for color in colors]

            plt.legend([0, 1, 2], ['EX', 'RL-S', 'RL-U'],
                       handler_map={
                           0: LegendObject(colors[0], colors_faded[0]),
                           1: LegendObject(colors[1], colors_faded[1]),
                           2: LegendObject(colors[2], colors_faded[2]),
                       })
            plt.xlabel("секунды")
            plt.title("" + m + ", " + d)
            plt.tight_layout()
            # plt.grid()

            plt.savefig('pics/fig_' + m + "_" + d)
            plt.clf()
            plt.close()
