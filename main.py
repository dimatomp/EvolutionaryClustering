from algorithms import *
from initialization import *
from mutations import *
from data_generation import *
from evaluation_indices import *
from log_handlers import *

if __name__ == "__main__":
    data, clusters = normalize_data(load_immunotherapy())  # generate_random_normal(2000, dim=2)
    index, sol = run_one_plus_one(axis_initialization, split_eliminate_mutation,
                                  silhouette_index, data,
                                  clusters, logging=Matplotlib2DLogger() if data.shape[1] == 2 else None)
    print('Resulting index value', index)
    print(list(sol["labels"]))
