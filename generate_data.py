from data_generation import *
import sys


def write_random_normal(dim, n_clusters, prefix):
    fname = prefix + '/' + generated_file_name(2000, dim=dim, n_clusters=n_clusters)
    data, labels = normalize_data(generate_random_normal(2000, dim=dim, n_clusters=n_clusters))
    with open(fname, 'w') as f:
        for point, cluster in zip(data, labels):
            print(*('{:.18e}'.format(p) for p in point), cluster, sep=',', file=f)


if __name__ == '__main__':
    prefix = '.' if len(sys.argv) == 1 else sys.argv[1]
    write_random_normal(2, 10, prefix)
    write_random_normal(2, 30, prefix)
    write_random_normal(10, 10, prefix)
    write_random_normal(10, 30, prefix)
