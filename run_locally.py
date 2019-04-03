from main import *

if __name__ == '__main__':
    run_one_plus_lambda_task('/dev/stdout', calinski_harabaz_index,
                             normalize_data(load_vehicle()), axis_initialization,
                             list(map(SingleMoveMutation, get_all_moves())),
                             # None)
                             lambda f: CSVLogger(output=f, log_unsuccessful=False))
