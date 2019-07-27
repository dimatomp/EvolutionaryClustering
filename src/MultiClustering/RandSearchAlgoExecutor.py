
import time
import traceback
import sys

from . import Constants
from . import ExThreads
from .ExThreads import LimitType
from .mab_solvers.MabSolver import TL


class RandSearchAlgoExecutor:
    def __init__(self, metric, X, seed, limit_type=LimitType.TIME):
        self.metric = metric
        self.X = X
        self.best_val = Constants.best_init
        self.best_param = dict()
        self.best_algo = "-1"
        self.seed = seed
        self.saved_parameters = dict()
        self.limit_type = limit_type
        self.per_algo = {}
        self.n = [0] * Constants.num_algos
        self.th = [0] * 7

    def apply(self, arm, file=None, iteration_number=1, remaining_budget=1):
        algo = self.th[arm]  # algo : ClusteringThread = th[i]
        lim = TL(remaining_budget)

        while True:
            if lim.is_limit_exceeded():
                break

            cfg = algo.config_space.sample_configuration()
            try:
                start = time.time()
                value = algo.target_run(cfg)

                if self.limit_type == LimitType.TIME:
                    lim.consume_limit(time.time() - start)
                elif self.limit_type == LimitType.CALLS:
                    lim.consume_limit(1)
                else:
                    raise "Unsupported Limit Type: " + str(self.limit_type)

                # It'd be unfair to record value if limit is over. skip it
                if lim.is_limit_exceeded():
                    break

                self.n[arm] += 1
                if value < algo.value:
                    algo.value = value
                    algo.parameters = cfg
            except:
                exc_info = sys.exc_info()
                print("Error occured while fitting " + algo.thread_name)
                print("Error occured while fitting " + algo.thread_name, file=sys.stderr)
                traceback.print_exception(*exc_info)
                del exc_info

        if file is not None:
            file.write( str(iteration_number) + ', ' + self.metric + ', '
                        + str(self.best_val) + ', ' + self.best_algo + ', '
                        + Constants.algos[arm] + ', ' + str(algo.value) + '\n')
            file.flush()

        return (Constants.in_reward - algo.value) / Constants.in_reward

    def execute(self, budget, f):
        # just shortcuts
        metric = self.metric
        X = self.X
        seed = self.seed
        num = Constants.num_algos
        budget = float(budget)

        self.th[0] = ExThreads.km_thread(metric, X, seed, int(budget / num), self.limit_type)
        self.th[1] = ExThreads.aff_thread(metric, X, seed, int(budget / num), self.limit_type)
        self.th[2] = ExThreads.ms_thread(metric, X, seed, int(budget / num), self.limit_type)
        self.th[3] = ExThreads.w_thread(metric, X, seed, int(budget / num), self.limit_type)
        self.th[4] = ExThreads.db_thread(metric, X, seed, int(budget / num), self.limit_type)
        self.th[5] = ExThreads.gm_thread(metric, X, seed, int(budget / num), self.limit_type)
        self.th[6] = ExThreads.bgm_thread(metric, X, seed, int(budget / num), self.limit_type)

        for i in range(0, num):
            self.apply(arm=i, file=f, remaining_budget=budget / num)

            if self.th[i].value < self.best_val:
                self.best_val = self.th[i].value
                self.best_algo = self.th[i].thread_name
                self.best_param = self.th[i].parameters

        return self.best_val

