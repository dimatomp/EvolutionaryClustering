datas = [
    ('generated_2dim_10cl', 'load_generated_random_normal(2000, dim=2, n_clusters=10, prefix=generated_prefix)'),
    ('generated_2dim_30cl', 'load_generated_random_normal(2000, dim=2, n_clusters=30, prefix=generated_prefix)'),
    ('generated_10dim_10cl', 'load_generated_random_normal(2000, dim=10, n_clusters=10, prefix=generated_prefix)'),
    ('generated_10dim_30cl', 'load_generated_random_normal(2000, dim=10, n_clusters=30, prefix=generated_prefix)'),
    ('iris', 'normalize_data(load_iris(prefix=real_prefix))'),
    ('immunotherapy', 'normalize_data(load_immunotherapy(prefix=real_prefix))'),
    ('user_knowledge', 'normalize_data(load_user_knowledge(prefix=real_prefix))'),
    ('mfeat_morphological', 'normalize_data(load_mfeat_morphological(prefix=real_prefix))'),
    ('glass', 'normalize_data(load_glass(prefix=real_prefix))'),
    ('haberman', 'normalize_data(load_haberman(prefix=real_prefix))'),
    ('heart_statlog', 'normalize_data(load_heart_statlog(prefix=real_prefix))'),
    ('vehicle', 'normalize_data(load_vehicle(prefix=real_prefix))'),
    ('liver_disorders', 'normalize_data(load_liver_disorders(prefix=real_prefix))'),
    ('oil_spill', 'normalize_data(load_oil_spill(prefix=real_prefix))')
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
mutations = [
    ## These mutations no longer exist
    # ('centroid_hill_climbing', 'centroid_initialization', 'centroid_hill_climbing_mutation'),
    # ('prototype_hill_climbing', 'prototype_initialization', 'prototype_hill_climbing_mutation'),
    # ('knn_reclassification', 'axis_initialization', 'knn_reclassification_mutation'),
    # ('one_nth_change', 'axis_initialization', 'one_nth_change_mutation')
    ## These mutations are obsolete
    # ('evocluster_diameter_centroid', 'axis_initialization',
    #  'evo_cluster_mutation(diameter_separation, centroid_distance_cohesion)'),
    # ('evocluster_mean_centroid', 'axis_initialization',
    #  'evo_cluster_mutation(mean_centroid_distance_separation, centroid_distance_cohesion)'),
    # ('evocluster_validity_centroid', 'axis_initialization',
    #  'evo_cluster_mutation(density_based_validity_separation, centroid_distance_cohesion)'),
    # ('evocluster_sparseness_centroid', 'axis_initialization',
    #  'evo_cluster_mutation(density_based_sparseness_separation, centroid_distance_cohesion)'),
    # ('evocluster_validity_separation', 'axis_initialization',
    #  'evo_cluster_mutation(density_based_validity_separation, density_based_separation_cohesion)'),
    # ('evocluster_sparseness_separation', 'axis_initialization',
    #  'evo_cluster_mutation(density_based_sparseness_separation, density_based_separation_cohesion)'),
    # ('split_eliminate', 'axis_initialization', 'split_eliminate_mutation'),
    # ('split_merge_move', 'axis_initialization', 'split_merge_move_mutation'),
    ('all_mutations_trivial', 'axis_initialization', 'all_moves_mutation(silent=True)'),
    ('all_mutations_dynamic', 'axis_initialization', 'all_moves_dynamic_mutation(silent=True)')
]


def get_file_name(index, data, mutation):
    return '{}-{}-{}.txt'.format(index, data, mutation)


# tasks = [('state-of-the-art.txt', 'run_state_of_the_art([datas, indices])')]
tasks = []
for mutation_name, init, mutation in mutations:
    for index_name, index in indices:
        for data_name, data in datas:
            fname = get_file_name(index_name, data_name, mutation_name)
            tasks.append((fname, "run_task(output_prefix + '/' + '{}', {}, {}, {}, {})".format(fname, index, data, init,
                                                                                               mutation)))