from algorithms import *
from initialization import *
from mutations import *
from data_generation import *
from evaluation_indices import *
from log_handlers import *
from multiprocessing import Pool
from sklearn.metrics import adjusted_rand_score


def run_task(args):
    fname, index, (data, clusters), mutation = args
    print('Launching', fname, file=sys.stderr)
    index = eval(index)
    with open(fname, 'w') as f:
        index, sol = run_one_plus_one(axis_initialization, mutation,
                                      index, data,
                                      lambda labels: adjusted_rand_score(clusters, labels['labels']),
                                      n_clusters=len(np.unique(clusters)),
                                      logging=default_logging(f))
        print('Resulting index value', index, file=f)
        print(list(sol["labels"]), file=f)


if __name__ == "__main__":
    datas = [
        ('generated_2dim_3cl', generate_random_normal(2000, dim=2, n_clusters=3)),
        ('generated_2dim_5cl', generate_random_normal(2000, dim=2, n_clusters=5)),
        ('generated_2dim_10cl', generate_random_normal(2000, dim=2, n_clusters=10)),
        ('generated_2dim_30cl', generate_random_normal(2000, dim=2, n_clusters=30)),
        ('generated_10dim_3cl', generate_random_normal(2000, dim=10, n_clusters=3)),
        ('generated_10dim_5cl', generate_random_normal(2000, dim=10, n_clusters=5)),
        ('generated_10dim_10cl', generate_random_normal(2000, dim=10, n_clusters=10)),
        ('generated_10dim_30cl', generate_random_normal(2000, dim=10, n_clusters=30)),
        ('iris', normalize_data(load_iris())),
        ('immunotherapy', normalize_data(load_immunotherapy())),
        ('user_knowledge', normalize_data(load_user_knowledge())),
    ]
    mutations = [
        ('split_merge_move', split_merge_move_mutation),
        ('split_eliminate', split_eliminate_mutation),
        ('one_nth_change', one_nth_change_mutation)
    ]
    indices = [
        ('silhouette', 'silhouette_index'),
        ('calinski_harabaz', 'calinski_harabaz_index'),
        ('davies_bouldin', 'davies_bouldin_index'),
        ('dvcb_2', 'dvcb_index()'),
        ('dvcb_5', 'dvcb_index(d=5)'),
        ('dunn', 'dunn_index'),
        ('generalized_dunn_41', 'generalized_dunn_index(separation="centroid_distance", cohension="diameter")'),
        ('generalized_dunn_43', 'generalized_dunn_index(separation="centroid_distance", cohension="mean_distance")'),
        ('generalized_dunn_51', 'generalized_dunn_index(separation="mean_per_cluster", cohension="diameter")'),
        ('generalized_dunn_53', 'generalized_dunn_index(separation="mean_per_cluster", cohension="mean_distance")'),
        ('generalized_dunn_13', 'generalized_dunn_index(separation="single_linkage", cohension="mean_distance")'),
        ('generalized_dunn_31', 'generalized_dunn_index(separation="mean_per_point", cohension="diameter")'),
        ('generalized_dunn_33', 'generalized_dunn_index(separation="mean_per_point", cohension="mean_distance")'),
    ]
    tasks = []
    for index_name, index in indices:
        eval(index)
        for data_name, data in datas:
            for mutation_name, mutation in mutations:
                tasks.append(('{}-{}-{}.txt'.format(index_name, data_name, mutation_name), index, data, mutation))
    with Pool(4) as pool:
        pool.map(run_task, tasks)
