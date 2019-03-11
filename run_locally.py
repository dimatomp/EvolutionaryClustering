from main import *

if __name__ == '__main__':
    run_task(['/dev/stdout', dvcb_index(2), normalize_data(generate_random_normal(2000, dim=2, n_clusters=10)),
              axis_initialization,
              all_moves_mutation('density_based_validity_separation', 'density_based_separation_cohesion')])
