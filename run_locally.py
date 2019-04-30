from main import *

if __name__ == '__main__':
    run_one_plus_lambda_task('/dev/stdout', silhouette_index,
                             normalize_data(load_from_file('svmguide3.csv', prefix='datasets/regular')), tree_initialization,
                             list(map(SingleMoveMutation, get_all_moves())),
                             #evo_cluster_mutation("mean_centroid_distance_separation", "centroid_distance_cohesion"),
                             #lambda f: CSVLogger(output=f, log_unsuccessful=False))
                             lambda f: Matplotlib2DLogger(CSVLogger(output=f, log_unsuccessful=False)))
