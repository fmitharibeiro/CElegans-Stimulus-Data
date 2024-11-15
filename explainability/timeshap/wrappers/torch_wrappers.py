#  Copyright 2022 Feedzai
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Wrapper to explain custom PyTorch RNN models.
"""

import numpy as np
import pandas as pd
import copy
import math
from typing import Tuple

import torch

from ...timeshap.wrappers import TimeSHAPWrapper


class TorchModelWrapper(TimeSHAPWrapper):
    """Wrapper for pytorch machine learning models.

    Encompasses necessary logic to utilize torch models as lambda functions
    required for TimeSHAP explanations.

    This wrapper is responsible to create torch tensors, sending them to the
    required device, batching processes, and obtained predictions from tensors.

    Attributes
    ----------
    model: torch.nn.Module
        Torch model to wrap. This model is required to receive a torch.Tensor
        of sequences and returning the score for each instance of each sequence.

    batch_budget: int
        The number of instances to score at a time. Needed to not overload
        GPU memory.
        Default is 750K. Equates to a 7GB batch

    device: torch.device

    Methods
    -------
    predict_last(data: pd.DataFrame, metadata: Matrix) -> list
        Creates explanations for each instance in ``data``.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 batch_budget: int = 750000,
                 device: torch.device = None,
                 batch_ignore_seq_len: bool = False,
                 ):
        super().__init__(model, batch_budget)
        self.batch_ignore_seq_len = batch_ignore_seq_len
        if device:
            self.model = self.model.to(device)

    def prepare_input(self, input):
        sequence = copy.deepcopy(input)
        if isinstance(sequence, pd.DataFrame):
            sequence = np.expand_dims(sequence.values, axis=0)
        elif len(sequence.shape) == 2 and isinstance(sequence, np.ndarray):
            sequence = np.expand_dims(sequence, axis=0)

        if not (len(sequence.shape) == 3 and isinstance(sequence, np.ndarray)):
            raise ValueError("Input type not supported")

        return sequence

    def predict_last_hs(self,
                     sequences: np.ndarray,
                     hidden_states: np.ndarray = None,
                     ) -> Tuple[np.ndarray, Tuple[np.ndarray]]:
        sequences = self.prepare_input(sequences)
        device = next(self.model.parameters()).device

        sequence_len = sequences.shape[1]
        batch_size = math.floor(self.batch_budget / sequence_len) if not self.batch_ignore_seq_len else self.batch_budget
        batch_size = max(1, batch_size)

        if sequences.shape[0] <= batch_size:
            with torch.no_grad():
                self.model.train(False)
                data_tensor = torch.from_numpy(sequences.copy()).float().to(device)
                if hidden_states is not None:
                    if isinstance(hidden_states, tuple):
                        if isinstance(hidden_states[0], tuple):
                            # for LSTM
                            hidden_states_tensor = tuple(tuple(torch.from_numpy(y).float().to(device)for y in x) for x in hidden_states)
                        else:
                            hidden_states_tensor = tuple(torch.from_numpy(x).float().to(device) for x in hidden_states)
                    else:
                        hidden_states_tensor = torch.from_numpy(hidden_states).float().to(device)
                    predictions = self.model(data_tensor, hidden_states_tensor)
                else:
                    predictions = self.model(data_tensor)
                self.model.train(True)
                if not isinstance(predictions, tuple):
                    if isinstance(predictions, torch.Tensor):
                        return predictions.cpu().numpy()
                    return predictions
                elif isinstance(predictions, tuple) and len(predictions) == 2:
                    predictions, hs = predictions
                    if isinstance(hs, tuple):
                        if isinstance(hs[0], tuple):
                            # for LSTM
                            return predictions.cpu().numpy(), tuple(tuple(y.cpu().numpy() for y in x) for x in hs)
                        else:
                            return predictions.cpu().numpy(), tuple(x.cpu().numpy() for x in hs)
                    else:
                        return predictions.cpu().numpy(), hs.cpu().numpy()
                else:
                    raise NotImplementedError(
                        "Only models that return predictions or predictions + hidden states are supported for now.")

        else:
            return_scores = []
            return_hs = []
            hs = None
            for i in range(0, sequences.shape[0], batch_size):
                batch = sequences[i:(i + batch_size), :, :]
                batch_tensor = torch.from_numpy(batch.copy())
                with torch.no_grad():
                    self.model.train(False)
                    batch_tensor = batch_tensor.float().to(device)
                    if hidden_states is not None:
                        if isinstance(hidden_states, tuple):
                            if isinstance(hidden_states[0], tuple):
                                # for LSTM
                                batch_hs_tensor = tuple(tuple(torch.from_numpy(y[:, i:(i + batch_size),:].copy()).float().to(device).float().to(device) for y in x) for x in hidden_states)
                            else:
                                batch_hs_tensor = tuple(torch.from_numpy(x[:, i:(i + batch_size), :].copy()).float().to(device).float().to(device) for x in hidden_states)
                        else:
                            batch_hs_tensor = torch.from_numpy(hidden_states[:, i:(i + batch_size), :].copy()).float().to(device)
                        predictions = self.model(batch_tensor, batch_hs_tensor)
                    else:
                        predictions = self.model(batch_tensor)
                    self.model.train(True)

                    if not isinstance(predictions, tuple):
                        if isinstance(predictions, torch.Tensor):
                            return predictions.cpu().numpy()
                        return predictions
                    elif isinstance(predictions, tuple) and len(predictions) == 2:
                        predictions, hs = predictions
                        predictions = predictions.cpu()
                        if isinstance(hs, tuple):
                            if isinstance(hs[0], tuple):
                                if return_hs == []:
                                    return_hs = [[[] for _ in x] for x in hs]
                                for ith, ith_layer_hs in enumerate(hs):
                                    return_hs[ith][0].append(ith_layer_hs[0].cpu().numpy())
                                    return_hs[ith][1].append(ith_layer_hs[1].cpu().numpy())
                            else:
                                if return_hs == []:
                                    return_hs = [[] for _ in hs]
                                for ith, ith_layer_hs in enumerate(hs):
                                    return_hs[ith].append(ith_layer_hs.cpu().numpy())
                        else:
                            hs = hs.cpu().numpy()
                            return_hs.append(hs)
                    else:
                        raise NotImplementedError(
                            "Only models that return predictions or predictions + hidden states are supported for now.")

                return_scores.append(predictions.numpy())
            if hs is None:
                return np.concatenate(tuple(return_scores), axis=0)

            if isinstance(hs, tuple):
                if isinstance(hs[0], tuple):
                    return_hs = tuple(tuple(np.concatenate(tuple(y), axis=1) for y in x) for x in return_hs)
                else:
                    return_hs = tuple(np.concatenate(tuple(x), axis=1) for x in return_hs)
            else:
                return_hs = np.concatenate(tuple(return_hs), axis=1)
            return np.concatenate(tuple(return_scores), axis=0), return_hs
