import numpy as np


class Individual:
    def __init__(self, data_fields):
        self.data_fields = data_fields
        self.partition_fields = dict()
        self.prev_partition_fields = None
        self.is_immutable = False
        self.callback = None

    def __getitem__(self, field):
        if field in self.data_fields:
            return self.data_fields[field]
        if field in self.partition_fields:
            return self.partition_fields[field]
        return self.prev_partition_fields[field]

    def __contains__(self, item):
        return item in self.data_fields or item in self.partition_fields or self.prev_partition_fields is not None and item in self.prev_partition_fields

    def receive_feedback(self, success, time):
        if self.callback is not None:
            self.callback(success, time)

    def set_callback(self, func):
        self.callback = func

    def set_data_field(self, key, value):
        if self.is_immutable:
            raise AttributeError("Trying to modify immutable individual")
        self.data_fields[key] = value

    def set_partition_field(self, key, value):
        if self.is_immutable:
            raise AttributeError("Trying to modify immutable individual")
        if key in self.partition_fields:
            raise AttributeError('Key "{}" already present in the inidividual'.format(key))
        self.prev_partition_fields = None
        self.partition_fields[key] = value

    def copy(self):
        self.is_immutable = True
        for v in self.partition_fields.values():
            if isinstance(v, np.ndarray):
                v.flags.writeable = False
        result = Individual(self.data_fields)
        result.prev_partition_fields = self.partition_fields
        return result
