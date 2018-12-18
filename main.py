from algorithms import *
from clustering_models import *
from data_generation import *
from evaluation_indices import *

if __name__ == "__main__":
    data, clusters = generate_random_normal(2000)
    print('Generated', data.shape, 'dataset with', clusters, 'clusters')
    index, sol = run_one_plus_one(SinglePointChangeModel(), silhouette_index, data, clusters, False)
    print('Resulting index value', index)
    print(list(sol[0]))
