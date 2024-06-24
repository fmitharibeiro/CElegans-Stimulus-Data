import numpy as np

from shap.utils._legacy import Data


class DenseData(Data):
    def __init__(self, data, group_names, *args):
        self.groups = args[0] if len(args) > 0 and args[0] is not None else [np.array([i]) for i in range(len(group_names))]

        j = sum(len(g) for g in self.groups)
        num_samples = data.shape[0]
        t = False
        if j != data.shape[2]:
            t = True
            num_samples = data.shape[2]

        valid = (not t and j == data.shape[2]) or (t and j == data.shape[0])
        if not valid:
            raise ValueError("# of names must match data matrix!")

        self.weights = args[1] if len(args) > 1 else np.ones(num_samples)
        self.weights /= np.sum(self.weights)
        wl = len(self.weights)
        valid = (not t and wl == data.shape[0]) or (t and wl == data.shape[1])
        if not valid:
            raise ValueError("# of weights must match data matrix!")

        self.transposed = t
        self.group_names = group_names
        self.data = data
        self.groups_size = len(self.groups)


def convert_to_data(val, keep_index=False):
    if isinstance(val, np.ndarray):
        return DenseData(val, [str(i) for i in range(val.shape[2])])
    else:
        raise NotImplementedError #Check original convert_to_data
    

def compute_background(X, method):
    if method == "feat_mean":
        return np.mean(X, axis=0)
    elif method == "zeros":
        return np.zeros(X.shape[1])
    elif method == "min":
        return np.min(X, axis=0)
    elif method == "median":
        return np.median(X, axis=0)
    else:
        raise NotImplementedError