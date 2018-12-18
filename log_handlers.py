import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors


def default_logging(n_step, c_index, c_solution, n_mutations, minor):
    print("[minor] " if minor else "", n_step, ": index value ", c_index, ', total ', n_mutations, ' mutations',
          sep='')


class Matplotlib2DLogger:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        plt.ion()
        plt.show(block=False)
        self.clusters = None

    def __call__(self, n_step, c_index, c_solution, n_mutations, minor):
        default_logging(n_step, c_index, c_solution, n_mutations, minor)
        values, data = c_solution
        self.ax.clear()
        n_clusters = len(np.unique(values))
        if n_clusters != self.clusters:
            print('Now there are', n_clusters, 'clusters')
            self.clusters = n_clusters
        for i in range(self.clusters):
            color = colors.hsv_to_rgb((i / self.clusters, 1, 1))[None, :]
            self.ax.scatter(*data[values == i].T, c=color, s=3)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
