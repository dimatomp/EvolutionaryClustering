from main import *

if __name__ == '__main__':
    run_one_plus_lambda_task('/dev/stdout', calinski_harabaz_index,
                          generate_random_normal(2000, dim=2, n_clusters=30),
                          tree_initialization,
                          list(map(SingleMoveMutation, get_all_moves())),
                          lambda f: CSVLogger(output=f, log_unsuccessful=False))
                          #lambda f: Matplotlib2DLogger(CSVLogger(output=f, log_unsuccessful=False)))
    #run_shalamov('/dev/stdout', calinski_harabaz_index, normalize_data(load_from_file('dataset_187_abalone.csv', prefix='datasets/regular')))
