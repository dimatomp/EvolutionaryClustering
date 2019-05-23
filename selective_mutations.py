from mutations import *
import pandas as pd
import pickle


def true_moves_mutation(dataset, index, **kwargs):
    with open('mutation_usage.dat', 'rb') as f:
        mutation_usage = pickle.load(f)
    fname = "{}-{}-one_plus_lambda_all_moves.txt".format(index, dataset)
    for name, usage in mutation_usage:
        if name == fname:
            return TrivialStrategyMutation(list(np.array(get_all_moves())[usage]), **kwargs)


def predicted_moves_mutation(dataset, index, prefix='.', **kwargs):
    with open('cv10.dat', 'rb') as f:
        cv10 = pickle.load(f)
    for i, s in enumerate(cv10):
        if dataset in s:
            frame = pd.read_csv(prefix + '/metaframe_{:02d}.csv'.format(i))
            with open(prefix + '/metapredicted_{:02d}.dat'.format(i), 'rb') as f:
                pred = pickle.load(f)
            index = ['silhouette', 'calinski_harabaz', 'davies_bouldin', 'dvcb', 'dunn', 'generalized_dunn_41',
                     'generalized_dunn_43', 'generalized_dunn_51', 'generalized_dunn_53', 'generalized_dunn_13'].index(
                index)
            row = np.argwhere((frame['index'] == index) & (frame['dataname'] == dataset))[0, 0]
            return TrivialStrategyMutation(list(np.array(get_all_moves())[pred[row].astype('bool')]), **kwargs)
