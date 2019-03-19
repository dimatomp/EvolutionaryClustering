from log_handlers import default_logging
from time import time
from numpy import isnan


def run_one_plus_one(initialization, mutation, index, data, external_measure, n_clusters=2, boundary=1e-9,
                     num_tries=1000, logging=None):
    c_solution = initialization(data, n_clusters)
    c_index = index(c_solution)
    ground_truth = external_measure(c_solution)
    logging(-1, c_index, c_solution, 0, ground_truth, False, True, 0, None)
    last_breakthrough = c_index

    def is_better(a, b):
        return a < b if index.is_minimized else a > b

    def last_boundary():
        return last_breakthrough - boundary if index.is_minimized else last_breakthrough + boundary

    logging = logging or default_logging()
    n_step = 0
    n_mutations = 0
    while True:
        at_least_one = False
        for i in range(num_tries):
            n_step += 1
            start = time()
            n_solution = mutation(c_solution)
            start = time() - start
            if isinstance(n_solution, tuple):
                n_solution, detail = n_solution
            else:
                detail = None
            n_index = index(n_solution)
            if isnan(n_index):
                raise ValueError("Index equal to NaN")
            success = is_better(n_index, c_index)
            n_solution.receive_feedback(success, start)
            if success:
                at_least_one = True
                n_mutations += 1
                c_solution, c_index = n_solution, n_index
                ground_truth = external_measure(c_solution)
                if isnan(ground_truth):
                    raise ValueError("External index equal to NaN")
                if is_better(n_index, last_boundary()):
                    last_breakthrough, c_tries = c_index, 0
                    logging(n_step, c_index, c_solution, n_mutations, ground_truth, False, True, start, detail)
                    continue
                else:
                    logging(n_step, c_index, c_solution, n_mutations, ground_truth, True, True, start, detail)
            else:
                logging(n_step, c_index, c_solution, n_mutations, ground_truth, False, False, start, detail)
        if at_least_one:
            mutation.recalibrate()
        else:
            break
    return c_index, c_solution
