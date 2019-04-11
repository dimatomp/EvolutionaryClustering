from main import *

if __name__ == '__main__':
    run_one_plus_lambda_task('/dev/stdout', calinski_harabaz_index,
                             normalize_data(generate_random_normal(2000, dim=2, n_clusters=30)), axis_initialization,
                             RandomMovesFromList(10, list(map(SingleMoveMutation, get_all_moves()))),
                             # None)
                             lambda f: CSVLogger(output=f, log_unsuccessful=False))
