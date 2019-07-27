from random import randrange

from src.MultiClustering.Constants import batch_size
from src.MultiClustering.RLrfAlgoEx import RLrfrsAlgoEx
from src.MultiClustering.mab_solvers.Smx_R import SoftmaxR
from src.algorithms import *
from src.batch_tasks import *
from src.log_handlers import *
from src.selective_mutations import *
from src.evaluation_indices import *
from src.initialization import *
from src.data_generation import *
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


def run_shalamov(fname, index, data, index_name, data_name):
    print('Launching', fname, file=sys.stderr)
    times = pd.read_csv('shalamov_running_times.csv', index_col=0)
    time_limit = times[(times['index'] == index_name) & (times['dataset'] == data_name)]['time_predicted_mean'].values
    assert time_limit.shape == (1,)
    time_limit = min(time_limit[0], 600)
    with open(fname, 'w') as f:
        algo_e = RLrfrsAlgoEx(index, data, randrange(2 ** 32), batch_size, expansion=100)
        mab_solver = SoftmaxR(action=algo_e, time_limit=time_limit)
        mab_solver.initialize(f, None)
        start = time()
        mab_solver.iterate(100000, f)
        start = time() - start
        print("Running time", start, file=f)
        print("Resulting index", mab_solver.action.best_val * (1 if index.is_minimized else -1), file=f)
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
