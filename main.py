from algorithms import *
from initialization import *
from mutations import *
from data_generation import *
from evaluation_indices import *
from log_handlers import *
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering, MeanShift
from time import time
import os
import traceback


def run_task(args):
    fname, index, (data, clusters), initialization, mutation = args
    print('Launching', fname, file=sys.stderr)
    with open(fname, 'w') as f:
        try:
            index = eval(index)
            mutation = eval(mutation)
            start = time()
            index, sol = run_one_plus_one(initialization, mutation,
                                          index, data,
                                          lambda labels: adjusted_rand_score(clusters, labels['labels']),
                                          n_clusters=len(np.unique(clusters)),
                                          logging=default_logging(f))
            print('Running time', time() - start, 'seconds', file=f)
            print('Resulting index value', index, file=f)
            print(list(sol["labels"]), file=f)
        except:
            traceback.print_exc(file=f)
    print('Finished', fname, file=sys.stderr)


if __name__ == "__main__":
    datas = [
        ('generated_2dim_10cl', normalize_data(generate_random_normal(2000, dim=2, n_clusters=10))),
        ('generated_2dim_30cl', normalize_data(generate_random_normal(2000, dim=2, n_clusters=30))),
        ('generated_10dim_10cl', normalize_data(generate_random_normal(2000, dim=10, n_clusters=10))),
        ('generated_10dim_30cl', normalize_data(generate_random_normal(2000, dim=10, n_clusters=30))),
        ('iris', normalize_data(load_iris())),
        ('immunotherapy', normalize_data(load_immunotherapy())),
        ('user_knowledge', normalize_data(load_user_knowledge())),
        ('mfeat_morphological', normalize_data(load_mfeat_morphological())),
        ('glass', normalize_data(load_glass())),
        ('haberman', normalize_data(load_haberman())),
        ('heart_statlog', normalize_data(load_heart_statlog())),
        ('vehicle', normalize_data(load_vehicle())),
        ('liver_disorders', normalize_data(load_liver_disorders())),
        ('oil_spill', normalize_data(load_oil_spill()))
    ]
    indices = [
        ('silhouette', 'silhouette_index'),
        ('calinski_harabaz', 'calinski_harabaz_index'),
        ('davies_bouldin', 'davies_bouldin_index'),
        ('dvcb_2', 'dvcb_index()'),
        # ('dvcb_5', 'dvcb_index(d=5)'),
        ('dunn', 'dunn_index'),
        ('generalized_dunn_41', 'generalized_dunn_index(separation="centroid_distance", cohesion="diameter")'),
        ('generalized_dunn_43', 'generalized_dunn_index(separation="centroid_distance", cohesion="mean_distance")'),
        ('generalized_dunn_51', 'generalized_dunn_index(separation="mean_per_cluster", cohesion="diameter")'),
        ('generalized_dunn_53', 'generalized_dunn_index(separation="mean_per_cluster", cohesion="mean_distance")'),
        ('generalized_dunn_13', 'generalized_dunn_index(separation="single_linkage", cohesion="mean_distance")'),
        # ('generalized_dunn_31', 'generalized_dunn_index(separation="mean_per_point", cohension="diameter")'),
        # ('generalized_dunn_33', 'generalized_dunn_index(separation="mean_per_point", cohension="mean_distance")'),
    ]
    if os.fork() == 0:
        print('Launching state-of-the-art.txt', file=sys.stderr)
        with open('state-of-the-art.txt', 'w') as f:
            for dataname, dataset in datas:
                data, clusters = dataset
                n_clusters = len(np.unique(data[1]))
                for modelname, model in [('KMeans', KMeans(n_clusters=n_clusters)),
                                         ('Affinity', AffinityPropagation()),
                                         ('AgglomerativeClustering', AgglomerativeClustering(n_clusters=n_clusters)),
                                         ('MeanShift', MeanShift())]:
                    start = time()
                    labels = model.fit_predict(data)
                    start = time() - start
                    for indexname, index in indices:
                        index = eval(index)
                        try:
                            print(dataname, modelname, indexname, adjusted_rand_score(clusters, labels),
                                  index({"labels": labels, "data": data}), start, len(np.unique(labels)), file=f)
                        except:
                            traceback.print_exc(file=f)
                    f.flush()
        print('Finished state-of-the-art.txt', file=sys.stderr)
        exit()
    mutations = [
        ('evocluster_diameter_centroid', axis_initialization, 'evo_cluster_mutation(diameter_separation, centroid_distance_cohesion)'),
        ('evocluster_mean_centroid', axis_initialization, 'evo_cluster_mutation(mean_centroid_distance_separation, centroid_distance_cohesion)'),
        ('evocluster_validity_centroid', axis_initialization, 'evo_cluster_mutation(density_based_validity_separation, centroid_distance_cohesion)'),
        ('evocluster_sparseness_centroid', axis_initialization, 'evo_cluster_mutation(density_based_sparseness_separation, centroid_distance_cohesion)'),
        ('centroid_hill_climbing', centroid_initialization, 'centroid_hill_climbing_mutation'),
        ('prototype_hill_climbing', prototype_initialization, 'prototype_hill_climbing_mutation'),
        ('evocluster_validity_separation', axis_initialization, 'evo_cluster_mutation(density_based_validity_separation, density_based_separation_cohesion)'),
        ('knn_reclassification', axis_initialization, 'knn_reclassification_mutation'),
        ('evocluster_sparseness_separation', axis_initialization, 'evo_cluster_mutation(density_based_sparseness_separation, density_based_separation_cohesion)'),
        ('split_eliminate', axis_initialization, 'split_eliminate_mutation'),
        ('split_merge_move', axis_initialization, 'split_merge_move_mutation'),
        ('one_nth_change', axis_initialization, 'one_nth_change_mutation')
    ]
    tasks = []
    for index_name, index in indices:
        eval(index)
        for data_name, data in datas:
            for mutation_name, init, mutation in mutations:
                tasks.append(('{}-{}-{}.txt'.format(index_name, data_name, mutation_name), index, data, init, mutation))
    for i, task in enumerate(tasks):
        if i >= 3:
            os.wait()
        if os.fork() == 0:
            run_task(task)
            exit()
    for i in range(4):
        os.wait()
