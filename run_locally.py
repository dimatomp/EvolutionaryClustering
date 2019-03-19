from main import *

if __name__ == '__main__':
    run_task('/dev/stdout', calinski_harabaz_index, normalize_data(generate_random_normal(2000, dim=2, n_clusters=30)),
             axis_initialization,
             all_moves_dynamic_mutation(), lambda f: csv_logging(output=f, log_unsuccessful=False))
