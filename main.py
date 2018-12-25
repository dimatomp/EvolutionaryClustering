from algorithms import *
from clustering_models import *
from data_generation import *
from evaluation_indices import *
from log_handlers import *

if __name__ == "__main__":
    data, clusters = generate_random_normal(20000, dim=2)
    index, sol = run_one_plus_one(SplitMergeMoveModel(), calinski_harabaz_index, data, clusters, minimize=False,
                                  logging=Matplotlib2DLogger())
    print('Resulting index value', index)
    print(list(sol[0]))
