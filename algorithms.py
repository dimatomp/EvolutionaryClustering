from time import time
from numpy import isnan
from .mutations import MutationNotApplicable
from random import choice
import traceback
import sys


def run_one_plus_one(initialization, mutation, index, data, logging, n_clusters=2, boundary=1e-9, num_tries=250):
    c_solution = initialization(data, n_clusters)
    c_index = index(c_solution)
    logging(-1, c_index, c_solution, 0, False, True, 0, None)
    last_breakthrough = c_index

    def is_better(a, b):
        return a < b if index.is_minimized else a > b

    def last_boundary():
        return last_breakthrough - boundary if index.is_minimized else last_breakthrough + boundary

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
                if is_better(n_index, last_boundary()):
                    last_breakthrough, c_tries = c_index, 0
                    logging(n_step, c_index, n_solution, n_mutations, False, True, start, detail)
                    continue
                else:
                    logging(n_step, c_index, n_solution, n_mutations, True, True, start, detail)
            else:
                logging(n_step, c_index, n_solution, n_mutations, False, False, start, detail)
        if at_least_one:
            mutation.recalibrate()
        else:
            break
    return c_index, c_solution


def run_one_plus_lambda(initialization, moves, index, data, logging, n_clusters=2, num_tries=250):
    c_solution = initialization(data, n_clusters)
    c_index = index(c_solution)
    logging(-1, c_index, c_solution, 0, False, True, 0, None)

    def is_better(a, b):
        return a < b if index.is_minimized else a > b

    n_step = 0
    n_mutations = 0
    while True:
        at_least_one = False
        for i in range(num_tries):
            n_step += 1
            mutation_indices = []
            times = []
            n_solutions = []
            details = []
            eval_indices = []

            def measure_time(i, mutation):
                try:
                    start = time()
                    n_solution = mutation(c_solution)
                    start = time() - start
                    n_solution, detail = n_solution if isinstance(n_solution, tuple) else (n_solution, None)
                    n_index = index(n_solution)
                    times.append(start)
                except MutationNotApplicable:
                    return
                except:
                    traceback.print_exc()
                    return
                if isnan(n_index):
                    print("Index equal to NaN for mutation " + i, file=sys.stderr)
                mutation_indices.append(i)
                n_solutions.append(n_solution)
                details.append(detail)
                eval_indices.append(n_index)

            for i, m in enumerate(moves):
                measure_time(i, m)

            successful_solution = None
            successful_index = None
            successful_indiv = None
            for i, n_solution, n_index in zip(mutation_indices, n_solutions, eval_indices):
                success = is_better(n_index, successful_index or c_index)
                if success:
                    successful_solution = i
                    successful_index = n_index
                    successful_indiv = n_solution
            for i, sol, timeV, indexV, detail in zip(mutation_indices, n_solutions, times, eval_indices, details):
                sol.receive_feedback(i == successful_solution, timeV)
                better_anyway = is_better(indexV, c_index)
                logging(n_step, indexV if better_anyway else c_index, sol, n_mutations, better_anyway,
                        i == successful_solution, timeV, detail)

            if successful_solution is not None:
                at_least_one = True
                n_mutations += 1
                c_solution, c_index = successful_indiv, successful_index
        if not at_least_one:
            break
    return c_index, c_solution


class RandomMovesFromList:
    def __init__(self, lam, moves):
        self.lam = lam
        self.moves = moves

    def __iter__(self):
        self.cIdx = 0
        return self

    def __next__(self):
        if self.cIdx >= self.lam:
            raise StopIteration
        self.cIdx += 1
        return choice(self.moves)
