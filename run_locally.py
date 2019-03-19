from main import *

if __name__ == '__main__':
    run_task('/dev/stdout', dvcb_index(2), normalize_data(load_haberman()),
             random_initialization,
             all_moves_mutation(), lambda f: csv_logging(output=f, log_unsuccessful=False))
