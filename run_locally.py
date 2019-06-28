from .main import *

if __name__ == '__main__':
    #run_one_plus_one_task('/dev/stdout', silhouette_index,
    #                      load_user_knowledge(prefix='datasets'),
    #                      tree_initialization,
    #                      all_moves_mutation(),
    #                      # evo_cluster_mutation("mean_centroid_distance_separation", "centroid_distance_cohesion"),
    #                      lambda f: CSVLogger(output=f, log_unsuccessful=False))
    #                      #lambda f: Matplotlib2DLogger(CSVLogger(output=f, log_unsuccessful=False)))
    run_shalamov('/dev/stdout', calinski_harabaz_index, normalize_data(load_from_file('dataset_187_abalone.csv', prefix='datasets/regular')))
