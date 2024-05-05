import numpy as np
from .segmentation import SeqShapSegmentation

class SeqShapKernel:
    def __init__(self, model, seq_num, dataset_name, background="feat_mean", random_seed=None, f=None):
        """
        Initialize the SeqShapKernel object. This class operates on a single sample.

        Parameters:
        - model: The machine learning model to be explained.
        - background: The background dataset used for reference.
        - random_seed: Random seed for reproducibility (optional).
        - f: A function like (#seqs, #feats) -> #seqs
        """
        self.model = model
        self.background = background
        self.random_seed = random_seed
        self.sequence_number = seq_num
        self.dataset_name = dataset_name
        self.k = 0

        if f:
            self.f = f
        else:
            self.f = lambda x: self.model.predict(x) # Default prediction function

    
    def __call__(self, X):
        ''' X shape: (#events, #feats)
        '''
        self.background = self.compute_background(X, self.background)

        seg = SeqShapSegmentation(self.f, self.sequence_number, self.dataset_name)

        segmented_X = seg(X)
        self.k = segmented_X.shape[0]
        pass

    def compute_background(self, X, method):
        if method == "feat_mean":
            return np.mean(X, axis=0)
        else:
            raise NotImplementedError

