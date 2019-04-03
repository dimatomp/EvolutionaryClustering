from main import *

if __name__ == '__main__':
    run_one_plus_lambda_task('/dev/stdout', dvcb_index(2),
                             normalize_data(load_vehicle()), axis_initialization,
                             list(map(SingleMoveMutation, get_all_moves())),
                             # None)
                             lambda f: csv_logging(output=f, log_unsuccessful=False))
