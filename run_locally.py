from main import *

if __name__ == '__main__':
    run_task('/dev/stdout', silhouette_index, normalize_data(generate_random_normal(2000, dim=2, n_clusters=10)),
             axis_initialization,
             all_moves_mutation(), lambda f: csv_logging(output=f, log_unsuccessful=False))
