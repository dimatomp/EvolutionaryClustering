import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib import colors


def default_logging(output=None):
    def logging(n_step, c_index, c_solution, n_mutations, ground_truth, minor):
        print("[minor] " if minor else "", n_step, ": index value ", c_index, ", mutual info ", ground_truth,
              ', total ',
              n_mutations, ' mutations, ', len(np.unique(c_solution["labels"])), ' clusters',
              sep='', file=output or sys.stdout)
        if output is not None:
            output.flush()

    return logging


class Matplotlib2DLogger:
    def __init__(self, nested=None):
        self.nested = nested
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        plt.ion()
        plt.show(block=False)

    def __call__(self, n_step, c_index, c_solution, n_mutations, ground_truth, minor):
        if self.nested is not None:
            self.nested(n_step, c_index, c_solution, n_mutations, ground_truth, minor)
        labels, data = c_solution["labels"], c_solution["data"]
        self.ax.clear()
        n_clusters = len(np.unique(labels))
        for i in range(n_clusters):
            color = colors.hsv_to_rgb((i / n_clusters, 1, 1))[None, :]
            self.ax.scatter(*data[labels == i].T, c=color, s=3)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
