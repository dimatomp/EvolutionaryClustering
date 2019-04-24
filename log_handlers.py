import numpy as np
import sys
from io import BytesIO
from base64 import b64decode, b64encode
from matplotlib import pyplot as plt
from matplotlib import colors
from time import time


def encode_solution(labels):
    labels = np.hstack([[-1], labels])
    inequalities = np.hstack([[False], labels[:-1] != labels[1:]])
    indices = (np.argwhere(inequalities).flatten() - 1).astype(np.int16)
    values = labels[inequalities].astype('int8')
    output = BytesIO()
    output.write(np.int16(len(labels) - 1))
    output.write(np.int16(len(values)))
    output.write(values)
    output.write(indices[1:])
    return b64encode(output.getvalue()).decode()


def decode_solution(string):
    input = BytesIO(b64decode(string.encode()))
    length, rle = np.frombuffer(input.read(4), count=2, dtype=np.int16)
    values = np.frombuffer(input.read(rle), count=rle, dtype=np.int8)
    indices = np.frombuffer(input.read((rle - 1) * 2), count=rle - 1, dtype=np.int16)
    indices = np.hstack([[0], indices, [length]])
    labels = np.zeros(length, dtype=np.int)
    for f, t, v in zip(indices, indices[1:], values):
        labels[f:t] = v
    return labels


class CSVLogger:
    def __init__(self, output=None, log_unsuccessful=False):
        self.output = output or sys.stdout
        self.log_unsuccessful = log_unsuccessful
        self.start_time = None
        self.prev_labels = None
        print('generation,index,ext_index,n_successful_mutations,n_clusters,delta_time,time,detail,individual',
              file=output or sys.stdout)

    def __call__(self, n_step, c_index, c_solution, n_mutations, ground_truth, minor, success, delta_time, detail):
        c_time = time()
        self.start_time = self.start_time or c_time
        if not self.log_unsuccessful and not success: return
        solution_diff = c_solution['labels'] - self.prev_labels if self.prev_labels is not None else c_solution[
            'labels']
        self.prev_labels = c_solution['labels']
        print(n_step, c_index, ground_truth, n_mutations, len(np.unique(c_solution["labels"])), delta_time,
              c_time - self.start_time, str(detail).replace(',', ';'), encode_solution(solution_diff), sep=',',
              file=self.output)
        self.output.flush()


class Matplotlib2DLogger:
    def __init__(self, nested=None, log_unsuccessful=False, fnames=None):
        self.log_unsuccessful = log_unsuccessful
        self.nested = nested
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.fnames = fnames
        self.findex = 0
        plt.ion()
        plt.show(block=False)

    def __call__(self, n_step, c_index, c_solution, n_mutations, ground_truth, minor, success, time, detail):
        if not self.log_unsuccessful and not success: return
        if self.nested is not None:
            self.nested(n_step, c_index, c_solution, n_mutations, ground_truth, minor, success, time, detail)
        labels, data = c_solution["labels"], c_solution["data"]
        self.ax.clear()
        n_clusters = len(np.unique(labels))
        for i in range(n_clusters):
            color = colors.hsv_to_rgb((i / n_clusters, 1, 1))[None, :]
            self.ax.scatter(*data[labels == i].T, c=color, s=3)
        self.fig.suptitle('Generation {}, index value {}'.format(n_step, c_index))
        self.fig.canvas.draw()
        if self.fnames is not None:
            self.fig.savefig(self.fnames.format(self.findex))
            self.findex += 1
        self.fig.canvas.flush_events()
