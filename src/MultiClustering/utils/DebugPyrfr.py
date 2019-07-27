# model : AbstractEPM
#             Model that implements train() and predict(). Will use a
#             :class:`~smac.epm.rf_with_instances.RandomForestWithInstances` if not set.
import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from ConfigSpace import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, Constant
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.optimizer.acquisition import EI
from smac.scenario.scenario import Scenario
from smac.utils.constants import MAXINT
from smac.utils.util_funcs import get_types

from .. import Constants
from .. import ExThreads
from ..RLthread import RLthread

num_run = np.random.randint(1234567980)
rng = np.random.RandomState(seed=num_run)


configspace = ConfigurationSpace()
algorithm = CategoricalHyperparameter("algorithm", ["auto", "full", "elkan"])
tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2)
n_clusters = UniformIntegerHyperparameter("n_clusters", 2, 15)
n_init = UniformIntegerHyperparameter("n_init", 2, 15)
max_iter = UniformIntegerHyperparameter("max_iter", 50, 1500)
verbose = Constant("verbose", 0)
configspace.add_hyperparameters([n_clusters, n_init, max_iter, tol, verbose, algorithm])


scenario = Scenario({"run_obj": "quality",
                     "cs": configspace,
                     "deterministic": "true",
                     "runcount-limit": 10
                     })

scenario.cs.seed(rng.randint(MAXINT))
# initial EPM
types, bounds = get_types(scenario.cs, scenario.feature_array)



model = RandomForestWithInstances(types=types, bounds=bounds,
                                  instance_features=scenario.feature_array,
                                  seed=rng.randint(MAXINT),
                                  pca_components=scenario.PCA_DIM)

acquisition_function = EI(model=model)

data = pd.read_csv(Constants.experiment_path + "iris.csv")
X = np.array(data, dtype=np.double)




t = RLthread(Constants.kmeans_algo, Constants.silhouette_metric, X, 42, 1)
t.run()

t.new_scenario(20)
t.run()

print( t.smac.solver.model.rf.get_largest_values_of_trees())
print("OK")

# model.train(np.ndarray([1, 1]), np.ndarray([1,2]))
# vs = model.rf.all_leaf_values([1,2,3])
