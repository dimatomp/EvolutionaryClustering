from algorithms import *
from log_handlers import *
from initialization import *
from mutations import *
from data_generation import *
from evaluation_indices import *
from batch_tasks import *
from log_handlers import *
from selective_mutations import *
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering, MeanShift
from time import time
import traceback

generated_prefix = '.'
real_prefix = '.'
output_prefix = '.'
predicted_prefix = '.'
test_mode = '--test' in sys.argv


def run_state_of_the_art(args):
    datas, indices = args
    print('Launching state-of-the-art.txt', file=sys.stderr)
    with open(output_prefix + '/state-of-the-art.txt', 'w') as f:
        for dataname, dataset in datas:
            data, clusters = eval(dataset)
            if test_mode:
                print('Loaded state-of-the-art.txt', file=f)
                return
            n_clusters = len(np.unique(data[1]))
            for modelname, model in [('KMeans', KMeans(n_clusters=n_clusters)),
                                     ('Affinity', AffinityPropagation()),
                                     ('AgglomerativeClustering', AgglomerativeClustering(n_clusters=n_clusters)),
                                     ('MeanShift', MeanShift())]:
                start = time()
                labels = model.fit_predict(data)
                start = time() - start
                for indexname, index in indices:
                    index = eval(index)
                    try:
                        print(dataname, modelname, indexname, adjusted_rand_score(clusters, labels),
                              index({"labels": labels, "data": data}), start, len(np.unique(labels)), file=f)
                    except:
                        traceback.print_exc(file=f)
                f.flush()
    print('Finished state-of-the-art.txt', file=sys.stderr)


def run_one_plus_one_task(fname, index, data, initialization, mutation, logging=None):
    print('Launching', fname, file=sys.stderr)
    with open(fname, 'w') as f:
        logging = logging(f) if logging is not None else CSVLogger(f, log_unsuccessful=True)
        if test_mode:
            print('Loaded', fname, file=f)
            return
        run_one_plus_one(initialization, mutation, index, data, logging=logging, n_clusters=int(np.cbrt(len(data))))
    print('Finished', fname, file=sys.stderr)


def run_one_plus_lambda_task(fname, index, data, initialization, moves, logging=None):
    print('Launching', fname, file=sys.stderr)
    with open(fname, 'w') as f:
        logging = logging(f) if logging is not None else CSVLogger(f, log_unsuccessful=True)
        if test_mode:
            print('Loaded', fname, file=f)
            return
        run_one_plus_lambda(initialization, moves, index, data, logging=logging, n_clusters=int(np.cbrt(len(data))))
    print('Finished', fname, file=sys.stderr)


if __name__ == "__main__":
    real_prefix = sys.argv[1]
    tasks = init_batch(real_prefix)
    if len(sys.argv) == 6:
        generated_prefix = sys.argv[2]
        output_prefix = sys.argv[3]
        predicted_prefix = sys.argv[4]
    if len(sys.argv) == 2:
        print('Total', len(tasks), 'tasks to run')
    elif len(sys.argv) == 3:
        print(tasks[int(sys.argv[2])][0])
    else:
        eval(tasks[int(sys.argv[5])][1])
