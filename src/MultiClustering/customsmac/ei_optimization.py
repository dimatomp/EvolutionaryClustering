import abc
import logging
import time
import numpy as np

from typing import Iterable, List, Union, Tuple, Optional

import sys
from smac.configspace import get_one_exchange_neighbourhood, \
    convert_configurations_to_array, Configuration, ConfigurationSpace
from smac.optimizer.ei_optimization import AcquisitionFunctionMaximizer, RandomSearch, LocalSearch, ChallengerList
from smac.runhistory.runhistory import RunHistory
from smac.stats.stats import Stats
from smac.optimizer.acquisition import AbstractAcquisitionFunction

__author__ = "Aaron Klein, Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Aaron Klein"
__email__ = "kleinaa@cs.uni-freiburg.de"
__version__ = "0.0.1"


class InterleavedLocalAndRandomSearch(AcquisitionFunctionMaximizer):
    """Implements SMAC's default acquisition function optimization.
    
    This optimizer performs local search from the previous best points 
    according, to the acquisition function, uses the acquisition function to 
    sort randomly sampled configurations and interleaves unsorted, randomly 
    sampled configurations in between.
    
    Parameters
    ----------
    acquisition_function : ~smac.optimizer.acquisition.AbstractAcquisitionFunction
        
    config_space : ~smac.configspace.ConfigurationSpace
    
    rng : np.random.RandomState or int, optional
    """
    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None,
    ):
        super().__init__(acquisition_function, config_space, rng)
        self.random_search = RandomSearch(
            acquisition_function, config_space, rng
        )
        self.local_search = LocalSearch(
            acquisition_function, config_space, rng
        )
        self.max_acq_value = sys.float_info.min

    def maximize(
            self,
            runhistory: RunHistory,
            stats: Stats,
            num_points: int,
            *args
    ) -> Iterable[Configuration]:
        next_configs_by_local_search = self.local_search._maximize(
            runhistory, stats, 10,
        )

        # Get configurations sorted by EI
        next_configs_by_random_search_sorted = self.random_search._maximize(
            runhistory,
            stats,
            num_points - len(next_configs_by_local_search),
            _sorted=True,
        )

        # Having the configurations from random search, sorted by their
        # acquisition function value is important for the first few iterations
        # of SMAC. As long as the random forest predicts constant value, we
        # want to use only random configurations. Having them at the begging of
        # the list ensures this (even after adding the configurations by local
        # search, and then sorting them)
        next_configs_by_acq_value = (
            next_configs_by_random_search_sorted
            + next_configs_by_local_search
        )
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        self.logger.debug(
            "First 10 acq func (origin) values of selected configurations: %s",
            str([[_[0], _[1].origin] for _ in next_configs_by_acq_value[:10]])
        )
        # store the max last expansion (challengers generation)
        self.max_acq_value = next_configs_by_acq_value[0][0]

        next_configs_by_acq_value = [_[1] for _ in next_configs_by_acq_value]

        challengers = ChallengerList(next_configs_by_acq_value,
                                     self.config_space)
        return challengers

    def _maximize(
            self,
            runhistory: RunHistory,
            stats: Stats,
            num_points: int
    ) -> Iterable[Tuple[float, Configuration]]:
        raise NotImplementedError()
