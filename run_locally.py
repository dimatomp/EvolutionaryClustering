from main import *

if __name__ == '__main__':
    run_one_plus_one_task('/dev/stdout', davies_bouldin_index,
                          load_generated_random_normal(2000, dim=2, n_clusters=30, prefix='scratch'),
                          tree_initialization,
                          true_moves_mutation('generated_2dim_10cl', 'davies_bouldin'),
                          # evo_cluster_mutation("mean_centroid_distance_separation", "centroid_distance_cohesion"),
                          # lambda f: CSVLogger(output=f, log_unsuccessful=False))
                          lambda f: Matplotlib2DLogger(CSVLogger(output=f, log_unsuccessful=False)))
