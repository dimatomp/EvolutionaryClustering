import os
import wget
import zipfile
import pickle
import traceback
from scipy.stats import entropy, skew, kurtosis
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering, MeanShift

from .algorithms import run_one_plus_one
from .data_generation import transform_dataset, normalize_data
from .initialization import tree_initialization
from .log_handlers import CSVLogger, silent_logging
from .mutations import TrivialStrategyMutation, get_all_moves
from .evaluation_indices import *


class AutoEvolutionaryClustering(BaseEstimator, ClusterMixin):
    all_indices = ['silhouette', 'calinski_harabaz', 'davies_bouldin', 'dvcb', 'dunn', 'generalized_dunn_41',
                   'generalized_dunn_43', 'generalized_dunn_51', 'generalized_dunn_53', 'generalized_dunn_13']

    def __init__(self, clustering_index, classifier_path='metaclassifier.zip', verbose=False):
        self.clustering_index_name = clustering_index
        self.clustering_index = eval(next(iter(filter(lambda t: t[0] == clustering_index, indices)))[1])
        classifier_path = os.path.abspath(classifier_path)
        if not os.path.isfile(classifier_path):
            print('Downloading metaclassifier...', file=sys.stderr)
            wget.download('https://dl.dropboxusercontent.com/s/42qado1qem9629f/metaclassifier.zip', classifier_path)
        with zipfile.ZipFile(classifier_path) as zip_file:
            with zip_file.open('metaclassifier.data') as f:
                self.classifier = pickle.load(f)
        self.verbose = verbose

    def fit_predict(self, X, y=None):
        state_of_the_art = ['KMeans(n_clusters=n_clusters)', 'AffinityPropagation()',
                            'AgglomerativeClustering(n_clusters=n_clusters)', 'MeanShift()']
        X = normalize_data(transform_dataset(X))
        meta_features = [self.all_indices.index(self.clustering_index_name), np.log2(len(X)), np.log2(X.shape[1])]
        meta_attrs = []
        meta_entropy = []
        cont_attrs = []
        for i, s in enumerate(X.T):
            uniq = np.unique(s)
            uniq.sort()
            if len(uniq) < len(s) * 0.3:
                meta_attrs.append(i)
                ints = np.searchsorted(uniq, s)
                meta_entropy.append(entropy(ints))
            else:
                cont_attrs.append(i)
        meta_features.append(len(meta_attrs))
        meta_features.append(sum(meta_entropy) / len(meta_entropy) if len(meta_entropy) > 0 else 0)
        if len(cont_attrs) > 0:
            corr = np.abs(np.corrcoef(X[:, cont_attrs], rowvar=False))
            meta_features.append(
                corr[np.arange(0, len(corr))[None, :] <= np.arange(0, len(corr))[:, None]].mean() if isinstance(
                    corr, np.ndarray) else corr)
            meta_features.append(np.mean(skew(X[:, cont_attrs])))
            meta_features.append(np.mean(kurtosis(X[:, cont_attrs])))
        else:
            meta_features += [0, 0, 0]
        dists = pdist(X)
        meta_features.append(dists.mean())
        meta_features.append(dists.var())
        meta_features.append(dists.std())
        z = np.abs(((dists - meta_features[-3]) / meta_features[-1]))
        meta_features.append(skew(dists))
        meta_features.append(kurtosis(dists))
        hist = np.histogram(dists, bins=9, range=(0, dists.max()))[0]
        assert len(hist) == 9
        meta_features += list(hist)
        meta_features.append((z < 1).sum())
        meta_features.append(((1 <= z) & (z < 2)).sum())
        meta_features.append(((2 <= z) & (z < 3)).sum())
        meta_features.append((3 <= z).sum())
        n_clusters = int(np.cbrt(len(X)))
        values = []
        fnum = 1
        for algo in state_of_the_art:
            try:
                cls = eval(algo).fit_predict(X)
            except:
                traceback.print_exc()
                cls = None
            indiv = Individual({'data': X, 'labels': cls})
            for _, idxstr in indices:
                try:
                    idx = eval(idxstr)(indiv)
                except:
                    if cls is not None:
                        traceback.print_exc()
                    idx = np.nan
                meta_features.append(idx)
                values.append(idx)
                fnum += 1
        hyperpar = self.classifier.predict(np.array([meta_features]))[0].astype('bool')
        selected_moves = list(np.array(get_all_moves())[hyperpar])
        if self.verbose:
            print('Training with small moves', selected_moves, file=sys.stderr)
        mutation = TrivialStrategyMutation(selected_moves, silent=self.verbose == 2)
        return run_one_plus_one(tree_initialization, mutation, self.clustering_index, X,
                                logging=silent_logging if not self.verbose else CSVLogger(sys.stderr))[1]['labels']
