import numpy as np


class IntegerEncodingModel:
    def generate_initial(self, data, n_clusters):
        return np.random.randint(low=0, high=n_clusters, size=len(data)), data


class SinglePointChangeModel(IntegerEncodingModel):
    def mutate(self, indiv):
        values, data = indiv
        cluster_sums = np.bincount(values)
        index_to_change = np.random.choice(np.argwhere(cluster_sums[values] > 1).flatten())
        n_value = np.random.randint(low=0, high=len(cluster_sums))
        if n_value >= values[index_to_change]:
            n_value += 1
        values = values.copy()
        values[index_to_change] = n_value
        return values, data


class OneNthChangeModel(IntegerEncodingModel):
    def __init__(self, add_new_clusters=True):
        self.add_new_clusters = add_new_clusters

    def mutate(self, indiv):
        values, data = indiv
        numbers_to_change = np.zeros(len(values), dtype='bool')
        numbers_to_change[np.random.randint(len(values))] = 1
        numbers_to_change[np.random.randint(len(values), size=len(values)) == 0] |= True
        n_clusters = len(np.unique(values)) + (1 if self.add_new_clusters else 0)
        values = values.copy()
        values[numbers_to_change] = np.random.randint(n_clusters, size=numbers_to_change.sum())
        empty = np.cumsum(np.bincount(values) == 0)
        values -= empty[values]
        return values, data
