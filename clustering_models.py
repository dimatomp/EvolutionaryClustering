import numpy as np


class SinglePointChangeModel:
    def generate_initial(self, data, n_clusters):
        return np.random.randint(low=0, high=n_clusters, size=len(data)), data

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
