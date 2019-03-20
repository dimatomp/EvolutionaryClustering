from main import *

if __name__ == '__main__':
    run_task('/dev/stdout', generalized_dunn_index(separation="single_linkage", cohesion="mean_distance"),
             normalize_data(load_liver_disorders()), axis_initialization, all_moves_dynamic_mutation(),
             lambda f: csv_logging(output=f, log_unsuccessful=False))
