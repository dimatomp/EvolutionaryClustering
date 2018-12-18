from algorithms import *
from clustering_models import *
from data_generation import *
from evaluation_indices import *
from log_handlers import *

if __name__ == "__main__":
    data, clusters = generate_random_normal(2000, dim=2)
    print('Generated', data.shape, 'dataset with', clusters, 'clusters')
    index, sol = run_one_plus_one(OneNthChangeModel(add_new_clusters=False), DynamicSilhouetteIndex(), data, clusters,
                                  False,
                                  logging=Matplotlib2DLogger())
    print('Resulting index value', index)
    print(list(sol[0]))
