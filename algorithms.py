from log_handlers import default_logging


def run_one_plus_one(initialization, mutation, index, data, external_measure, n_clusters=2, boundary=1e-9,
                     num_tries=10000, logging=None):
    c_solution = initialization(data, n_clusters)
    c_index = index(c_solution)
    last_breakthrough, c_tries = c_index, 0

    def is_better(a, b):
        return a < b if index.is_minimized else a > b

    def last_boundary():
        return last_breakthrough - boundary if index.is_minimized else last_breakthrough + boundary

    logging = logging or default_logging()
    n_step = 0
    n_mutations = 0
    while c_tries < num_tries:
        n_step += 1
        n_solution = mutation(c_solution)
        n_index = index(n_solution)
        if is_better(n_index, c_index):
            n_mutations += 1
            c_solution, c_index = n_solution, n_index
            ground_truth = external_measure(c_solution)
            if is_better(n_index, last_boundary()):
                last_breakthrough, c_tries = c_index, 0
                logging(n_step, c_index, c_solution, n_mutations, ground_truth, False)
                continue
            else:
                logging(n_step, c_index, c_solution, n_mutations, ground_truth, True)
        c_tries += 1
    return c_index, c_solution
