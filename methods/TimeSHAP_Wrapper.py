import numpy as np
from typing import Tuple

from timeshap.wrappers import TimeSHAPWrapper


class KerasModelWrapper(TimeSHAPWrapper):
    """Wrapper for Keras Sequential models.

    Encompasses necessary logic to utilize Keras Sequential models as lambda functions
    required for TimeSHAP explanations.

    This wrapper is responsible for creating numpy arrays, batching processes,
    and obtaining predictions from the model.

    Attributes
    ----------
    model: keras.models.Sequential
        Keras Sequential model to wrap. This model is required to receive a numpy.ndarray
        of sequences and return the score for each instance of each sequence.

    batch_budget: int
        The number of instances to score at a time. Needed to not overload
        GPU memory.
        Default is 750K. Equates to a 7GB batch
    """

    def __init__(self, model=None, seed=33, batch_budget: int = 750000):
        super().__init__(model, batch_budget)
        self.seed = seed # Does nothing
        self.param_grid = {}

    def prepare_input(self, input):
        sequence = np.copy(input)
        if len(sequence.shape) == 2:
            sequence = np.expand_dims(sequence, axis=0)
        if not (len(sequence.shape) == 3 and isinstance(sequence, np.ndarray)):
            raise ValueError("Input type not supported")
        return sequence

    def predict_last_hs(self, sequences: np.ndarray, *args) -> Tuple[np.ndarray, Tuple[np.ndarray]]:
        sequences = self.prepare_input(sequences)

        sequence_len = sequences.shape[1]
        batch_size = self.batch_budget // sequence_len

        if sequences.shape[0] <= batch_size:
            predictions, hs = self.model.predict(sequences, return_state=True)
            if isinstance(hs, list):
                hs = tuple(h for h in hs)
            return predictions, hs

        return_scores = []
        return_hs = []
        for i in range(0, sequences.shape[0], batch_size):
            batch = sequences[i:(i + batch_size), :, :]
            batch_predictions, batch_hs = self.model.predict(batch, return_state=True)
            return_scores.append(batch_predictions)
            if isinstance(batch_hs, list):
                if return_hs == []:
                    return_hs = [[] for _ in batch_hs]
                for ith, ith_layer_hs in enumerate(batch_hs):
                    return_hs[ith].append(ith_layer_hs)
            else:
                if return_hs == []:
                    return_hs = [batch_hs]
                else:
                    return_hs.append(batch_hs)

        if isinstance(return_hs[0], list):
            return_hs = tuple(np.concatenate(hs_list, axis=0) for hs_list in return_hs)
        else:
            return_hs = np.concatenate(return_hs, axis=0)

        return np.concatenate(tuple(return_scores), axis=0), return_hs
    
    def set_params(self, **params):
        if not params:
            return self
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self

    def get_params(self):
        return {attr: getattr(self, attr)
                for attr in dir(self)
                if not callable(getattr(self, attr)) and not attr.startswith("__")}