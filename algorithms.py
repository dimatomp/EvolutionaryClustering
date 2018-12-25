import numpy as np
from log_handlers import default_logging
from sklearn.metrics import adjusted_mutual_info_score


def run_one_plus_one(model, index, data, true_labels, minimize=True, boundary=1e-9, num_tries=10000, logging=None):
    c_solution = model.generate_initial(data, len(np.unique(true_labels)))
    c_index = index(c_solution)
    last_breakthrough, c_tries = c_index, 0

    def is_better(a, b):
        return a < b if minimize else a > b

    def last_boundary():
        return last_breakthrough - boundary if minimize else last_breakthrough + boundary

    logging = logging or default_logging
    n_step = 0
    n_mutations = 0
    while c_tries < num_tries:
        n_step += 1
        n_solution = model.mutate(c_solution)
        n_index = index(n_solution)
        if is_better(n_index, c_index):
            n_mutations += 1
            c_solution, c_index = n_solution, n_index
            ground_truth = adjusted_mutual_info_score(true_labels, c_solution[0])
            if is_better(n_index, last_boundary()):
                last_breakthrough, c_tries = c_index, 0
                logging(n_step, c_index, c_solution, n_mutations, ground_truth, False)
                continue
            else:
                logging(n_step, c_index, c_solution, n_mutations, ground_truth, True)
        c_tries += 1
    return c_index, c_solution
