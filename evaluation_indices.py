from sklearn.metrics import silhouette_score, calinski_harabaz_score, davies_bouldin_score


def silhouette_index(indiv):
    values, data = indiv
    return silhouette_score(data, values)


def calinski_harabaz_index(indiv):
    values, data = indiv
    return calinski_harabaz_score(data, values)


def davies_bouldin_index(indiv):
    values, data = indiv
    return davies_bouldin_score(data, values)


'''
class DynamicIndex:
    def __init__(self, index):
        self.index = index
        self.prev_values = None

    def __call__(self, indiv):
        values, data = indiv
        n_clusters = len(np.unique(values))
        if self.prev_values is None:
            self.result = self.index.find(data, values, n_clusters)
        else:
            for idx in np.argwhere(self.prev_values != values).flatten():
                self.result = self.index.update(data, n_clusters, values, self.prev_values[idx], values[idx], idx)
        self.prev_values = values
        return self.result


class DynamicVRCIndex(DynamicIndex):
    def __init__(self):
        super(DynamicVRCIndex, self).__init__(VRCIndex())


class DynamicSilhouetteIndex(DynamicIndex):
    def __init__(self):
        super(DynamicSilhouetteIndex, self).__init__(SilhouetteIndex())
'''
