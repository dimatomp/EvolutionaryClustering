from scipy.spatial.distance import squareform
from sklearn.neighbors import BallTree
from cluster_measures import *


def one_nth_change_mutation(indiv: dict) -> dict:
    labels, data = indiv["labels"], indiv["data"]
    numbers_to_change = np.zeros(len(labels), dtype='bool')
    numbers_to_change[np.random.randint(len(labels))] = 1
    numbers_to_change[np.random.randint(len(labels), size=len(labels)) == 0] |= True
    cluster_sizes = np.bincount(labels)
    n_clusters = len(cluster_sizes)
    if np.random.randint(n_clusters ** 2) == 0:
        n_clusters += 1
    labels = labels.copy()
    labels[numbers_to_change] = np.random.randint(n_clusters, size=np.count_nonzero(numbers_to_change))
    cleanup_empty_clusters(labels)
    indiv = indiv.copy()
    indiv["labels"] = labels
    return indiv


def split_merge_move_mutation(indiv: dict) -> dict:
    labels, data = indiv["labels"], indiv["data"]
    cluster_sizes = np.bincount(labels)
    labels = labels.copy()
    while True:
        method = np.random.randint(3)
        if method == 0:
            ex_cluster = np.random.choice(np.argwhere(cluster_sizes > 1).flatten())
            centroid = data[labels == ex_cluster].mean(axis=0)
            norm = np.random.multivariate_normal(np.zeros(len(centroid)), np.identity(len(centroid)))
            dots = (data[labels == ex_cluster] - centroid).dot(norm)
            ratio = np.random.uniform(dots.min(), dots.max())
            negativeDot = dots < ratio
            negativeSum = np.count_nonzero(negativeDot)
            if negativeSum == 0 or negativeSum == len(negativeDot):
                continue
            labels[labels == ex_cluster] = np.where(negativeDot, len(cluster_sizes), ex_cluster)
        elif method == 1 and len(cluster_sizes) != 2:
            centroids = np.array([data[labels == i].mean(axis=0) for i in range(len(cluster_sizes))])
            dists = pdist(centroids, 'minkowski', p=1)
            dists = construct_probabilities(dists)
            pair = np.random.choice(len(dists), p=dists)
            dists = np.zeros(len(dists))
            dists[pair] = 1
            src_cluster, dst_cluster = np.argwhere(squareform(dists) == 1)[0]
            # src_cluster, dst_cluster = np.random.choice(len(cluster_sizes), 2, replace=False)
            labels[labels == src_cluster] = dst_cluster
            labels[labels > src_cluster] -= 1
        else:
            # src_cluster = np.random.randint(len(cluster_sizes))
            dst_cluster = np.random.randint(len(cluster_sizes))  # - 1)
            # if dst_cluster >= src_cluster:
            #    dst_cluster += 1
            centroid = data[labels == dst_cluster].mean(axis=0)
            otherElems = labels != dst_cluster  # == src_cluster
            dists = np.linalg.norm(data[otherElems] - centroid, ord=1, axis=1)
            dists = construct_probabilities(dists)
            n_points = np.count_nonzero(dists)
            n_points = np.random.binomial(n_points - 1, 1 / (n_points - 1)) + 1
            if n_points == np.count_nonzero(otherElems):
                continue
            n_indices = np.random.choice(np.argwhere(otherElems).flatten(), n_points, replace=False, p=dists)
            labels[n_indices] = dst_cluster
            cleanup_empty_clusters(labels)
        break
    indiv = indiv.copy()
    indiv["labels"] = labels
    return indiv


def split_eliminate_mutation(indiv: dict) -> dict:
    labels, data = indiv["labels"], indiv["data"]
    cluster_sizes = np.bincount(labels)
    labels = labels.copy()
    while True:
        method = np.random.randint(3)
        if method == 0:
            ex_cluster = np.random.choice(np.argwhere(cluster_sizes > 1).flatten())
            indices = np.argwhere(labels == ex_cluster).flatten()
            centroid = data[indices].mean(axis=0)
            centroid_dists = np.linalg.norm(data[indices] - centroid, axis=1)
            farthest = data[indices[np.argmax(centroid_dists)]]
            farthest_dists = np.linalg.norm(data[indices] - farthest, axis=1)
            n_cluster = farthest_dists < centroid_dists
            n_cluster_cnt = np.count_nonzero(n_cluster)
            if n_cluster_cnt == 0 or n_cluster_cnt == len(cluster_sizes):
                continue
            labels[labels == ex_cluster] = np.where(n_cluster, len(cluster_sizes), ex_cluster)
        elif method == 1 and len(cluster_sizes) > 2:
            centroids = np.array([data[labels == i].mean(axis=0) for i in range(len(cluster_sizes))])
            ex_cluster = np.random.randint(len(cluster_sizes))
            indices = np.argwhere(labels == ex_cluster).flatten()
            centroids = centroids[np.arange(0, len(cluster_sizes)) != ex_cluster]
            neighbours = np.argmin(cdist(data[indices], centroids), axis=1)
            neighbours[neighbours >= ex_cluster] += 1
            labels[indices] = neighbours
            labels[labels > ex_cluster] -= 1
        else:
            dst_cluster = np.random.randint(len(cluster_sizes))
            centroid = data[labels == dst_cluster].mean(axis=0)
            otherElems = labels != dst_cluster
            dists = np.linalg.norm(data[otherElems] - centroid, ord=1, axis=1)
            dists = construct_probabilities(dists)
            n_points = np.count_nonzero(dists)
            n_points = np.random.binomial(n_points - 1, 1 / (n_points - 1)) + 1 if n_points > 1 else 1
            if n_points == np.count_nonzero(otherElems):
                continue
            n_indices = np.random.choice(np.argwhere(otherElems).flatten(), n_points, replace=False, p=dists)
            labels[n_indices] = dst_cluster
            cleanup_empty_clusters(labels)
        break
    indiv = indiv.copy()
    indiv["labels"] = labels
    return indiv


def evo_cluster_mutation(separation, cohesion):
    def mutation(indiv: dict) -> tuple:
        labels, data = indiv["labels"], indiv["data"]
        clusters = np.unique(labels)
        cluster_sizes = np.count_nonzero(labels[:, None] == clusters[None, :], axis=0)
        labels_backup = labels
        labels = labels.copy()
        while True:
            guided = np.random.randint(2)
            method = np.random.randint(3) if len(clusters) > 2 else np.random.randint(2) * 2
            if method == 1:  # merge-gene
                if guided == 0:
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
                    detail = 'Unguided merge of {} pairs of clusters'.format(len(first_part_clusters))
                    assert labels.min() >= 0
                else:
                    n_chosen_pairs = len(clusters) // 2
                    if n_chosen_pairs > 1:
                        n_chosen_pairs = np.random.binomial(n_chosen_pairs - 1, 1 / (n_chosen_pairs - 1)) + 1
                    cohesions = cohesion(labels=labels, data=data, indiv=indiv, cluster_labels=clusters, ord=1)
                    cohesions = squareform(cohesions)
                    detail = 'Guided merge of {} pairs of clusters'.format(n_chosen_pairs)
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
            else:
                if guided == 0:
                    n_chosen_clusters = np.random.binomial(len(cluster_sizes) - 1, 1 / (len(cluster_sizes) - 1)) + 1
                    chosen_clusters = np.random.choice(clusters, n_chosen_clusters, replace=False)
                else:
                    measures = separation(labels=labels, data=data, indiv=indiv, cluster_labels=clusters, ord=1)
                    measures = construct_probabilities(measures, the_less_the_better=False)
                    n_clusters = np.count_nonzero(measures)
                    if method == 0:
                        n_clusters = min(n_clusters, len(cluster_sizes) - 1)
                    n_chosen_clusters = 1 if n_clusters == 1 else np.random.binomial(n_clusters - 1,
                                                                                     1 / (n_clusters - 1)) + 1
                    chosen_clusters = np.random.choice(clusters, n_chosen_clusters, replace=False, p=measures)
                if method == 0:  # remove-and-reclassify
                    chosen_cluster_labels = chosen_clusters[:, None] == labels[None, :]
                    chosen_cluster_counts = np.count_nonzero(chosen_cluster_labels, axis=1)
                    n_chosen_samples = np.random.binomial(chosen_cluster_counts - 1, np.where(chosen_cluster_counts > 1,
                                                                                              1 / (
                                                                                                      chosen_cluster_counts - 1),
                                                                                              1)) + 1
                    indices = None
                    for chosen, samples in zip(chosen_cluster_labels, n_chosen_samples):
                        choice = np.random.choice(np.argwhere(chosen).flatten(), samples, replace=False)
                        indices = np.concatenate((indices, choice)) if indices is not None else choice
                    if guided == 0:
                        labels[indices] = np.random.choice(clusters, len(indices))
                        detail = 'Unguided remove-and-reclassify of {} entries'.format(len(indices))
                    else:
                        other_clusters = clusters[np.isin(clusters, chosen_clusters, invert=True)]
                        centroids = np.array([data[labels == i].mean(axis=0) for i in other_clusters])
                        centroid_dists = cdist(data[indices], centroids, metric='minkowski')
                        # TODO How about randomness?
                        labels[indices] = other_clusters[centroid_dists.argmin(axis=1)]
                        detail = 'Guided remove-and-reclassify of {} entries'.format(len(indices))
                    if len(np.unique(labels)) == 1:
                        labels = labels_backup
                        continue
                else:  # split-gene
                    # TODO Unroll "for" into numpy serial ops?
                    for i, ex_cluster in enumerate(chosen_clusters):
                        centroid = data[labels == ex_cluster].mean(axis=0)
                        norm = np.random.multivariate_normal(np.zeros(len(centroid)), np.identity(len(centroid)))
                        dots = data[labels == ex_cluster].dot(norm)
                        if guided == 0:
                            ratio = np.random.uniform(dots.min(), dots.max())
                        else:
                            hist, bins = np.histogram(dots, int(np.ceil(np.sqrt(len(dots)))))
                            probs = construct_probabilities(hist)
                            index = np.random.choice(len(bins) - 1, p=probs)
                            ratio = np.random.uniform(bins[index], bins[index + 1])
                        negativeDot = dots < ratio
                        labels[labels == ex_cluster] = np.where(negativeDot, len(cluster_sizes) + i, ex_cluster)
                    detail = '{} split of {} clusters'.format('Guided' if guided == 1 else 'Unguided',
                                                              len(chosen_clusters))
            break
        cleanup_empty_clusters(labels)
        indiv = indiv.copy()
        indiv["labels"] = labels
        return indiv, detail

    return mutation


def centroid_hill_climbing_mutation(indiv: dict) -> tuple:
    while True:
        centroids, data = indiv['centroids'], indiv['data']
        method = np.random.choice(3)
        if method == 0:
            shape = centroids.shape
            centroids = centroids.flatten()
            delta = np.random.uniform(-1, 1)
            index = np.random.choice(len(centroids))
            centroids[index] = 2 * delta if centroids[index] == 0 else (1 + 2 * delta) * centroids[index]
            centroids = centroids.reshape(shape)
            detail = 'Alter centroid by delta {}'.format(delta)
        elif method == 1 and len(centroids) > 2:
            centroids = np.delete(centroids, np.random.choice(len(centroids)), axis=0)
            detail = 'Remove centroid'
        else:
            mindata, maxdata = data.min(axis=0), data.max(axis=0)
            centroids = np.concatenate(
                [centroids, [np.random.sample(centroids.shape[1]) * (maxdata - mindata) + mindata]])
            detail = 'Add centroid'
        labels, centroids = get_labels_by_centroids(centroids, data)
        if labels.max() == 0:
            continue
        indiv = indiv.copy()
        indiv['centroids'] = centroids
        indiv['labels'] = labels
        return indiv, detail


def prototype_hill_climbing_mutation(indiv: dict) -> tuple:
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
    indiv = indiv.copy()
    indiv['prototypes'] = prototypes
    indiv['labels'] = get_labels_by_prototypes(prototypes, data)
    return indiv, "Add {} and remove {} prototypes".format(n_add, n_remove)


def knn_reclassification_mutation(indiv: dict) -> tuple:
    if 'tree' in indiv:
        ball_tree = indiv['tree']
    else:
        ball_tree = BallTree(indiv['data'])
        indiv['tree'] = ball_tree
    data, labels = indiv['data'], indiv['labels']
    clusters = np.unique(labels)
    labels_backup, labels = labels, labels.copy()
    while True:
        method = np.random.randint(3)
        if method >= 1 or len(clusters) == 2:
            n_centers = np.random.binomial(len(data) - 1, 1 / (len(data) - 1)) + 1
            indices_to_move = np.random.choice(len(data), n_centers, replace=False)
            g = np.random.randint(1, len(data) // len(clusters))
            indices_to_move = ball_tree.query(data[indices_to_move], k=g)[1]
            n_clusters = len(clusters)
            if method == 2:
                n_clusters += 1
            for i, row in enumerate(indices_to_move):
                labels[row] = np.random.choice(n_clusters) if i != 0 or method == 1 else n_clusters - 1
            detail = 'Move {} centers, {} entries in total, g={}'.format(n_centers,
                                                                         len(np.unique(indices_to_move.flatten())), g)
        else:
            f, t = np.random.choice(clusters, 2, replace=False)
            labels[labels == f] = t
            detail = 'Merge clusters'
        cleanup_empty_clusters(labels)
        if len(np.unique(labels)) == 1:
            labels = labels_backup
            continue
        indiv = indiv.copy()
        indiv['labels'] = labels
        return indiv, detail
