from main import *

if __name__ == '__main__':
    run_one_plus_one_task('/dev/stdout', dvcb_index(),
                             normalize_data(generate_random_normal(2000, dim=2, n_clusters=30)), random_initialization,
                             all_moves_mutation(),
                             #evo_cluster_mutation("mean_centroid_distance_separation", "centroid_distance_cohesion"),
                             lambda f: Matplotlib2DLogger(CSVLogger(output=f, log_unsuccessful=False)))
