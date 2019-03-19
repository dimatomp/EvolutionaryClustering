from sklearn.neighbors import BallTree
from cluster_measures import *
from itertools import chain
from individual import Individual
import sys
import traceback


class MutationNotApplicable(Exception): pass


def copy_labels(indiv: Individual) -> tuple:
    labels, data = indiv['labels'], indiv['data']
    labels = labels.copy()
    indiv.set_partition_field('labels', labels)
    return labels, data


def one_nth_change_move(indiv: Individual) -> str:
    labels, data = copy_labels(indiv)
    n_entries = np.random.binomial(len(labels) - 1, 1 / (len(labels) - 1)) + 1
    numbers_to_change = np.random.choice(len(labels), n_entries, replace=False)
    n_clusters = len(np.unique(labels))
    adding_cluster = np.random.randint(n_clusters ** 2) == 0
    if adding_cluster:
        n_clusters += 1
    labels[numbers_to_change] = np.random.randint(n_clusters, size=n_entries)
    return "Reclassify {} entries".format(n_entries) + (
        '' if not adding_cluster or max(labels[numbers_to_change]) != n_clusters - 1 else ', add new cluster')


class DynamicStrategyMutation:
    def __init__(self, strategies, silent=False):
        self.strategy_names = strategies
        self.strategies = list(map(eval, strategies))
        self.silent = silent
        self.probs = np.array([1 / len(strategies)] * len(strategies))
        self.success_array = [1] * len(strategies)
        print('Init dynamic strategy with moves:', self.strategy_names, file=sys.stderr)

    def recalibrate(self):
        self.probs = construct_probabilities(np.array(self.success_array), the_less_the_better=False)
        print('New probabilities', self.probs, file=sys.stderr)
        self.success_array = [1] * len(self.strategies)

    def __call__(self, indiv: Individual):
        indiv_backup, indiv = indiv, indiv.copy()
        while True:
            strategy_index = np.random.choice(len(self.strategy_names), p=self.probs)
            try:
                detail = "{}: {}".format(self.strategy_names[strategy_index], self.strategies[strategy_index](indiv))
                cleanup_empty_clusters(indiv['labels'])
            except MutationNotApplicable:
                if not self.silent:
                    print('Mutation {} not applicable'.format(self.strategy_names[strategy_index]), file=sys.stderr)
                indiv = indiv_backup.copy()
                continue
            except:
                traceback.print_exc()
                indiv = indiv_backup.copy()
                continue
            if len(np.unique(indiv['labels'])) == 1:
                if not self.silent:
                    print('Tried to leave single cluster with whole dataset', file=sys.stderr)
                indiv = indiv_backup.copy()
                continue

            def callback(success, time):
                if success:
                    self.success_array[strategy_index] += 1

            indiv.set_callback(callback)
            break
        return indiv, detail


class TrivialStrategyMutation:
    def __init__(self, strategies, silent=False):
        self.strategy_names = strategies
        self.strategies = list(map(eval, strategies))
        self.silent = silent

    def recalibrate(self):
        pass

    def __call__(self, indiv: Individual):
        indiv_backup, indiv = indiv, indiv.copy()
        while True:
            strategy_index = np.random.choice(len(self.strategy_names))
            try:
                detail = "{}: {}".format(self.strategy_names[strategy_index], self.strategies[strategy_index](indiv))
                cleanup_empty_clusters(indiv['labels'])
            except MutationNotApplicable:
                if not self.silent:
                    print('Mutation {} not applicable'.format(self.strategy_names[strategy_index]), file=sys.stderr)
                indiv = indiv_backup.copy()
                continue
            except:
                traceback.print_exc()
                indiv = indiv_backup.copy()
                continue
            if len(np.unique(indiv['labels'])) == 1:
                if not self.silent:
                    print('Tried to leave single cluster with whole dataset', file=sys.stderr)
                indiv = indiv_backup.copy()
                continue
            break
        return indiv, detail


def expand_cluster_move(indiv: Individual) -> str:
    labels, data = copy_labels(indiv)
    clusters = np.unique(labels)
    dst_clusters = choose_clusters_unguided(clusters)
    total_n_points = 0
    n_chosen_clusters = choose_n_clusters(len(clusters))
    # TODO Unroll the loop with numpy?
    for i in range(n_chosen_clusters):
        if len(clusters) == 1:
            raise MutationNotApplicable
        dst_cluster = np.random.choice(clusters)
        centroid = data[labels == dst_cluster].mean(axis=0)
        otherElems = labels != dst_cluster  # == src_cluster
        dists = np.linalg.norm(data[otherElems] - centroid, ord=1, axis=1)
        dists = construct_probabilities(dists)
        n_points = np.count_nonzero(dists)
        n_points = (np.random.binomial(n_points - 1, 1 / (n_points - 1)) if n_points > 1 else 0) + 1
        n_indices = np.random.choice(np.argwhere(otherElems).flatten(), n_points, replace=False, p=dists)
        labels[n_indices] = dst_cluster
        total_n_points += n_points
        clusters = np.unique(labels)
    return 'Expand {} clusters with {} entries'.format(len(dst_clusters), total_n_points)


def split_farthest_move(indiv: Individual) -> str:
    labels, data = copy_labels(indiv)
    clusters = np.unique(labels)
    ex_clusters = choose_clusters_unguided(clusters)
    # TODO Unroll with numpy?
    for i, ex_cluster in enumerate(ex_clusters):
        indices = np.argwhere(labels == ex_cluster).flatten()
        centroid = data[indices].mean(axis=0)
        centroid_dists = np.linalg.norm(data[indices] - centroid, axis=1)
        farthest = data[indices[np.argmax(centroid_dists)]]
        farthest_dists = np.linalg.norm(data[indices] - farthest, axis=1)
        n_cluster = farthest_dists < centroid_dists
        labels[labels == ex_cluster] = np.where(n_cluster, len(clusters) + i, ex_cluster)
    return "Split {} clusters".format(len(ex_clusters))


def eliminate_move_body(indiv: Individual, ex_clusters: np.ndarray) -> str:
    labels, data = copy_labels(indiv)
    for ex_cluster in ex_clusters:
        indices = np.argwhere(labels == ex_cluster).flatten()
        clusters, centroids = get_clusters_and_centroids(labels, data)
        centroids = centroids[np.arange(0, len(clusters)) != ex_cluster]
        neighbours = np.argmin(cdist(data[indices], centroids), axis=1)
        neighbours[neighbours >= ex_cluster] += 1
        labels[indices] = neighbours
        labels[labels > ex_cluster] -= 1
    return 'Eliminate {} clusters'.format(len(ex_clusters))


def unguided_eliminate_move(indiv: dict) -> str:
    return eliminate_move_body(indiv, choose_clusters_unguided(np.unique(indiv['labels']), can_choose_all=False))


def guided_eliminate_move(separation):
    def mutation(indiv) -> str:
        labels = indiv['labels']
        clusters = np.unique(labels)
        cluster_sizes = np.count_nonzero(labels[:, None] == clusters[None, :], axis=0)
        return eliminate_move_body(indiv, choose_clusters_guided(labels, indiv['data'], indiv, clusters, separation,
                                                                 cluster_sizes, False))

    return mutation


def unguided_merge_gene_move(indiv: Individual) -> str:
    labels, data = copy_labels(indiv)
    clusters = np.unique(labels)
    if len(clusters) == 2:
        raise MutationNotApplicable
    cluster_sizes = np.count_nonzero(labels[:, None] == clusters[None, :], axis=0)
    n_chosen_clusters = np.random.binomial(len(cluster_sizes) - 2, 1 / (len(cluster_sizes) - 2)) + 2
    chosen_clusters = np.random.choice(clusters, n_chosen_clusters, replace=False)
    first_part_clusters = np.random.choice(chosen_clusters, len(chosen_clusters) // 2)
    second_part_clusters = chosen_clusters[np.isin(chosen_clusters, first_part_clusters, invert=True)]
    second_part_clusters = np.random.choice(second_part_clusters, len(first_part_clusters),
                                            replace=False)
    indices_to_replace = np.isin(labels, second_part_clusters)
    mapping = -np.ones(second_part_clusters.max() + 1, dtype='int')
    mapping[second_part_clusters] = first_part_clusters
    labels[indices_to_replace] = mapping[labels[indices_to_replace]]
    assert labels.min() >= 0
    return 'Merge {} pairs of clusters'.format(len(first_part_clusters))


def guided_merge_gene_move(cohesion):
    def mutation(indiv: Individual) -> str:
        labels, data = copy_labels(indiv)
        clusters = np.unique(labels)
        n_chosen_pairs = len(clusters) // 2
        if n_chosen_pairs > 1:
            n_chosen_pairs = np.random.binomial(n_chosen_pairs - 1, 1 / (n_chosen_pairs - 1)) + 1
        cohesions = cohesion(labels=labels, data=data, indiv=indiv, cluster_labels=clusters, ord=1)
        cohesions = squareform(cohesions)
        for i in range(n_chosen_pairs):
            lin_cohesion = squareform(cohesions)
            lin_cohesion = construct_probabilities(lin_cohesion)
            chosen_pair = np.random.choice(len(lin_cohesion), p=lin_cohesion)
            # TODO No smarter way to do this?
            bitarray = np.zeros(len(lin_cohesion), dtype='bool')
            bitarray[chosen_pair] = True
            bitarray = squareform(bitarray)
            indices = np.argwhere(bitarray)[0]
            labels[labels == clusters[indices.max()]] = clusters[indices.min()]
            labels[labels > clusters[indices.max()]] -= 1
            cohesions = np.delete(cohesions, indices.max(), axis=0)
            cohesions = np.delete(cohesions, indices.max(), axis=1)
        return 'Merge {} pairs of clusters'.format(n_chosen_pairs)

    return mutation


def choose_n_clusters(n_clusters, can_choose_all=True):
    diff = n_clusters - 1 if can_choose_all else n_clusters - 2
    return (np.random.binomial(diff, 1 / diff) if diff > 0 else 0) + 1


def choose_clusters_unguided(clusters, can_choose_all=True):
    return np.random.choice(clusters, choose_n_clusters(len(clusters), can_choose_all=can_choose_all), replace=False)


def unguided_split_gene_move(indiv: Individual) -> str:
    labels, data = copy_labels(indiv)
    clusters = np.unique(labels)
    cluster_sizes = np.count_nonzero(labels[:, None] == clusters[None, :], axis=0)
    chosen_clusters = choose_clusters_unguided(clusters)
    for i, ex_cluster in enumerate(chosen_clusters):
        centroid = data[labels == ex_cluster].mean(axis=0)
        norm = np.random.multivariate_normal(np.zeros(len(centroid)), np.identity(len(centroid)))
        dots = data[labels == ex_cluster].dot(norm)
        ratio = np.random.uniform(dots.min(), dots.max())
        negativeDot = dots < ratio
        labels[labels == ex_cluster] = np.where(negativeDot, len(cluster_sizes) + i, ex_cluster)
    return 'Split {} clusters'.format(len(chosen_clusters))


def unguided_remove_and_reclassify_move(indiv: Individual) -> str:
    labels, data = copy_labels(indiv)
    clusters = np.unique(labels)
    chosen_clusters = choose_clusters_unguided(clusters)
    chosen_cluster_labels = chosen_clusters[:, None] == labels[None, :]
    chosen_cluster_counts = np.count_nonzero(chosen_cluster_labels, axis=1)
    n_chosen_samples = np.random.binomial(chosen_cluster_counts - 1,
                                          np.where(chosen_cluster_counts > 1, 1 / (chosen_cluster_counts - 1), 1)) + 1
    indices = None
    for chosen, samples in zip(chosen_cluster_labels, n_chosen_samples):
        choice = np.random.choice(np.argwhere(chosen).flatten(), samples, replace=False)
        indices = np.concatenate((indices, choice)) if indices is not None else choice
    labels[indices] = np.random.choice(clusters, len(indices))
    return 'Remove and reclassify {} entries'.format(len(indices))


def choose_clusters_guided(labels, data, indiv, clusters, separation, cluster_sizes, can_choose_all):
    measures = separation(labels=labels, data=data, indiv=indiv, cluster_labels=clusters, ord=1)
    measures = construct_probabilities(measures, the_less_the_better=False)
    n_clusters = np.count_nonzero(measures)
    if not can_choose_all:
        n_clusters = min(n_clusters, len(cluster_sizes) - 1)
    n_chosen_clusters = 1 if n_clusters == 1 else np.random.binomial(n_clusters - 1,
                                                                     1 / (n_clusters - 1)) + 1
    return np.random.choice(clusters, n_chosen_clusters, replace=False, p=measures)


def guided_split_gene_move(separation):
    def mutation(indiv: Individual) -> str:
        labels, data = copy_labels(indiv)
        clusters = np.unique(labels)
        cluster_sizes = np.count_nonzero(labels[:, None] == clusters[None, :], axis=0)
        chosen_clusters = choose_clusters_guided(labels, data, indiv, clusters, separation, cluster_sizes, True)
        for i, ex_cluster in enumerate(chosen_clusters):
            centroid = data[labels == ex_cluster].mean(axis=0)
            norm = np.random.multivariate_normal(np.zeros(len(centroid)), np.identity(len(centroid)))
            dots = data[labels == ex_cluster].dot(norm)
            hist, bins = np.histogram(dots, int(np.ceil(np.sqrt(len(dots)))))
            probs = construct_probabilities(hist)
            index = np.random.choice(len(bins) - 1, p=probs)
            ratio = np.random.uniform(bins[index], bins[index + 1])
            negativeDot = dots < ratio
            labels[labels == ex_cluster] = np.where(negativeDot, len(cluster_sizes) + i, ex_cluster)
        return 'Split {} clusters'.format(len(chosen_clusters))

    return mutation


def guided_remove_and_reclassify_move(separation):
    def mutation(indiv: Individual) -> str:
        labels, data = copy_labels(indiv)
        clusters = np.unique(labels)
        cluster_sizes = np.count_nonzero(labels[:, None] == clusters[None, :], axis=0)
        chosen_clusters = choose_clusters_guided(labels, data, indiv, clusters, separation, cluster_sizes, False)
        chosen_cluster_labels = chosen_clusters[:, None] == labels[None, :]
        chosen_cluster_counts = np.count_nonzero(chosen_cluster_labels, axis=1)
        n_chosen_samples = np.random.binomial(chosen_cluster_counts - 1,
                                              np.where(chosen_cluster_counts > 1, 1 / (chosen_cluster_counts - 1),
                                                       1)) + 1
        indices = None
        for chosen, samples in zip(chosen_cluster_labels, n_chosen_samples):
            choice = np.random.choice(np.argwhere(chosen).flatten(), samples, replace=False)
            indices = np.concatenate((indices, choice)) if indices is not None else choice
        other_clusters = clusters[np.isin(clusters, chosen_clusters, invert=True)]
        centroids = np.array([data[labels == i].mean(axis=0) for i in other_clusters])
        centroid_dists = cdist(data[indices], centroids, metric='minkowski')
        # TODO How about randomness?
        labels[indices] = other_clusters[centroid_dists.argmin(axis=1)]
        return 'Remove and reclassify of {} entries'.format(len(indices))

    return mutation


def centroid_hill_climbing_move(indiv: Individual) -> str:
    if 'centroids' not in indiv:
        data = indiv['data']
        centroids = get_clusters_and_centroids(indiv['labels'], data)[1]
        labels, centroids = get_labels_by_centroids(centroids, data)
        indiv.set_partition_field('centroids', centroids)
        indiv.set_partition_field('labels', labels)
        return 'Initialize centroids'
    centroids, data = indiv['centroids'], indiv['data']
    centroids = centroids.copy()
    shape = centroids.shape
    centroids = centroids.flatten()
    delta = np.random.uniform(-1, 1)
    index = np.random.choice(len(centroids))
    centroids[index] = 2 * delta if centroids[index] == 0 else (1 + 2 * delta) * centroids[index]
    centroids = centroids.reshape(shape)
    labels, centroids = get_labels_by_centroids(centroids, data)
    indiv.set_partition_field('labels', labels)
    indiv.set_partition_field('centroids', centroids)
    return 'Alter centroid by delta {}'.format(delta)


def prototype_hill_climbing_move(indiv: Individual) -> str:
    if 'prototypes' not in indiv:
        data = indiv['data']
        clusters, centroids = get_clusters_and_centroids(indiv['labels'], data)
        prototype_indices = cdist(data, centroids).argmin(axis=0)
        prototypes = np.isin(np.arange(len(data)), prototype_indices)
        indiv.set_partition_field('prototypes', prototypes)
        indiv.set_partition_field('labels', get_labels_by_prototypes(prototypes, data))
        return 'Initialize prototypes'
    prototypes, data = indiv['prototypes'], indiv['data']
    prototypes = prototypes.copy()
    n_prototypes = np.count_nonzero(prototypes)
    n_remove = np.random.binomial(n_prototypes, 1 / n_prototypes)
    n_add = np.random.binomial(len(prototypes) - n_prototypes, 1 / (len(prototypes) - n_prototypes))
    n_remove = min(n_remove, n_prototypes + n_add - 2)
    if n_add > 0:
        prototypes[np.random.choice(np.argwhere(~prototypes).flatten(), n_add, replace=False)] = True
    if n_remove > 0:
        prototypes[np.random.choice(np.argwhere(prototypes).flatten(), n_remove, replace=False)] = False
    indiv.set_partition_field('prototypes', prototypes)
    indiv.set_partition_field('labels', get_labels_by_prototypes(prototypes, data))
    return "Add {} and remove {} prototypes".format(n_add, n_remove)


def knn_reclassification_move(indiv: Individual) -> str:
    if 'tree' in indiv:
        ball_tree = indiv['tree']
    else:
        ball_tree = BallTree(indiv['data'])
        indiv.set_data_field('tree', ball_tree)
    labels, data = copy_labels(indiv)
    method = np.random.randint(2)
    clusters = np.unique(labels)
    n_centers = np.random.binomial(len(data) - 1, 1 / (len(data) - 1)) + 1
    indices_to_move = np.random.choice(len(data), n_centers, replace=False)
    g = np.random.randint(1, len(data) // len(clusters))
    indices_to_move = ball_tree.query(data[indices_to_move], k=g)[1]
    n_clusters = len(clusters)
    if method == 1:
        n_clusters += 1
    for i, row in enumerate(indices_to_move):
        labels[row] = np.random.choice(n_clusters) if i != 0 or method == 1 else n_clusters - 1
    detail = 'Move {} centers, {} entries in total, g={}'.format(n_centers,
                                                                 len(np.unique(indices_to_move.flatten())), g)
    return detail


split_merge_move_mutation = TrivialStrategyMutation(
    ['unguided_split_gene_move', 'guided_merge_gene_move(centroid_distance_cohesion)', 'expand_cluster_move'])
split_eliminate_mutation = TrivialStrategyMutation(
    ['split_farthest_move', 'unguided_eliminate_move', 'expand_cluster_move'])


def evo_cluster_mutation(separation, cohesion):
    return TrivialStrategyMutation(['unguided_merge_gene_move', 'guided_merge_gene_move({})'.format(cohesion),
                                    'unguided_remove_and_reclassify_move', 'unguided_split_gene_move',
                                    'guided_remove_and_reclassify_move({})'.format(separation),
                                    'guided_split_gene_move({})'.format(separation)])


cohesion_move_names = ['guided_merge_gene_move']
all_cohesion_moves = list(chain(*(['{}({})'.format(j, i) for i in all_cohesions] for j in cohesion_move_names)))
separation_move_names = ['guided_remove_and_reclassify_move', 'guided_split_gene_move', 'guided_eliminate_move']
all_separation_moves = list(chain(*(['{}({})'.format(j, i) for i in all_separations] for j in separation_move_names)))


def all_moves_mutation(separation=None, cohesion=None, silent=False):
    separation_moves = all_separation_moves if separation is None else ['{}({})'.format(j, separation) for j in
                                                                        separation_move_names]
    cohesion_moves = all_cohesion_moves if cohesion is None else ['{}({})'.format(j, cohesion) for j in
                                                                  cohesion_move_names]
    return TrivialStrategyMutation(
        ['unguided_merge_gene_move', 'unguided_remove_and_reclassify_move', 'unguided_split_gene_move',
         'expand_cluster_move', 'split_farthest_move', 'unguided_eliminate_move', 'knn_reclassification_move',
         'one_nth_change_move', 'centroid_hill_climbing_move', 'prototype_hill_climbing_move'] + separation_moves +
        cohesion_moves, silent=silent)


def non_prototype_moves_dynamic_mutation(separation=None, cohesion=None, silent=False):
    separation_moves = all_separation_moves if separation is None else ['{}({})'.format(j, separation) for j in
                                                                        separation_move_names]
    cohesion_moves = all_cohesion_moves if cohesion is None else ['{}({})'.format(j, cohesion) for j in
                                                                  cohesion_move_names]
    return DynamicStrategyMutation(
        ['unguided_merge_gene_move', 'unguided_remove_and_reclassify_move', 'unguided_split_gene_move',
         'expand_cluster_move', 'split_farthest_move', 'unguided_eliminate_move', 'knn_reclassification_move',
         'one_nth_change_move'] + separation_moves + cohesion_moves, silent=silent)


def all_moves_dynamic_mutation(separation=None, cohesion=None, silent=False):
    separation_moves = all_separation_moves if separation is None else ['{}({})'.format(j, separation) for j in
                                                                        separation_move_names]
    cohesion_moves = all_cohesion_moves if cohesion is None else ['{}({})'.format(j, cohesion) for j in
                                                                  cohesion_move_names]
    return DynamicStrategyMutation(
        ['unguided_merge_gene_move', 'unguided_remove_and_reclassify_move', 'unguided_split_gene_move',
         'expand_cluster_move', 'split_farthest_move', 'unguided_eliminate_move', 'knn_reclassification_move',
         'one_nth_change_move', 'centroid_hill_climbing_move', 'prototype_hill_climbing_move'] + separation_moves +
        cohesion_moves, silent=silent)
