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

from typing import Callable, List, Union
import numpy as np
import pandas as pd
from ...timeshap.explainer.kernel import TimeShapKernel
import os
import re
import math
from pathlib import Path
from ...timeshap.utils import convert_to_indexes, convert_data_to_3d
from ...timeshap.explainer import temp_coalition_pruning
from ...timeshap.utils import get_tolerances_to_test
from ...timeshap.explainer.extra import save_multiple_files, read_multiple_files, \
    file_exists, detect_last_saved_file_index, count_rows_in_last_file, correct_shap_vals_format


def event_level(f: Callable,
                data: np.array,
                baseline: Union[np.ndarray, pd.DataFrame],
                pruned_idx: np.array,
                random_seed: int,
                nsamples: int,
                display_events: List[str] = None,
                path = None,
                verbose = False
                ) -> pd.DataFrame:
    """Method to calculate event level explanations

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: np.array
        Sequence to explain.

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.
        The baseline can be an average event or an average sequence

    pruned_idx: int
        Index to prune the sequence. All events up to this index are grouped

    random_seed: int
        Used random seed for the sampling process.

    nsamples: int
        The number of coalitions for TimeSHAP to sample.

    display_events: List[str]
        In-order list of event names to be displayed

    Returns
    -------
    pd.DataFrame
    """
    explainer = TimeShapKernel(f, baseline, random_seed, "event", path=path)
    shap_values = explainer.shap_values(data, pruning_idx=pruned_idx, nsamples=nsamples, verbose=verbose)

    if display_events is None:
        display_events = ["Event {}".format(str(int(i))) for i in -len(pruned_idx) + np.where(pruned_idx == 1)[0][::-1]]
    else:
        display_events = display_events[-len(shap_values)+1:]

    if np.any(pruned_idx == 0):
        display_events += ["Pruned Events"]

    ret_data = []
    for exp, event in zip(shap_values, display_events):
        rounded_exp = [0 if np.isclose(n, 0, atol=0.0) else n for n in exp]
        ret_data += [[random_seed, nsamples, event, rounded_exp]]
    return pd.DataFrame(ret_data, columns=['Random seed', 'NSamples', 'Feature', 'Shapley Value'])


def local_event(f: Callable[[np.ndarray], np.ndarray],
                data: np.array,
                event_dict: dict,
                entity_uuid: Union[str, int, float],
                entity_col: str,
                baseline: Union[pd.DataFrame, np.array],
                pruned_idx: np.array,
                verbose=False
                ) -> pd.DataFrame:
    """Method to calculate event level explanations or load them if path is provided

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: np.array
        Sequence to explain.

    event_dict: dict
        Information required for the event level explanation calculation

    entity_uuid: Union[str, int, float]
        The indentifier of the sequence that is being pruned.
        Used when fetching information from a csv of explanations

    entity_col: str
        Entity column to identify sequences

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.
        The baseline can be an average event or an average sequence

    pruned_idx: int
        Index to prune the sequence. All events up to this index are grouped

    Returns
    -------
    pd.DataFrame
    """
    if event_dict.get("path") is None or not os.path.exists(event_dict.get("path")):
        event_data = event_level(f, data, baseline, pruned_idx, event_dict.get("rs"), event_dict.get("nsamples"), path=event_dict.get("path"), verbose=verbose)
        if event_dict.get("path") is not None:
            # create directory
            if '/' in event_dict.get("path"):
                Path(event_dict.get("path").rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
            event_data.to_csv(event_dict.get("path"), index=False)
    elif event_dict.get("path") is not None and os.path.exists(event_dict.get("path")):
        event_data = pd.read_csv(event_dict.get("path"))
        if len(event_data.columns) == 5 and entity_col is not None:
            event_data = event_data[event_data[entity_col] == entity_uuid]
        elif len(event_data.columns) == 4:
            pass
        else:
            raise ValueError
    else:
        raise ValueError

    return event_data


def verify_event_dict(event_dict: dict):
    """Verifies the format of the event dict for event level explanations

    Parameters
    ----------
    event_dict: dict
    """
    if event_dict.get('path'):
        assert isinstance(event_dict.get('path'), str), "Provided path must be a string"

    if event_dict.get('rs', False):
        if isinstance(event_dict.get('rs'), int):
            event_dict['rs'] = [event_dict.get('rs')]
        elif isinstance(event_dict.get('rs'), list):
            assert np.array([isinstance(x, int) for x in event_dict.get('rs')]).all(), "All provided random seeds must be ints."
        else:
            raise ValueError("Unsuported format of random seeds(s). Please provide one seed or a list of them.")
    else:
        print("No random seed provided for event-level explanations. Using default: 42")
        event_dict['rs'] = [42]

    if event_dict.get('nsamples', False):
        if isinstance(event_dict.get('nsamples'), int):
            event_dict['nsamples'] = [event_dict.get('nsamples')]
        elif isinstance(event_dict.get('nsamples'), list):
            assert np.array([isinstance(x, int) for x in event_dict.get('nsamples')]).all(), "All provided nsamples must be ints."
        else:
            raise ValueError("Unsuported format of nsamples. Please provide value or a list of them.")
    else:
        print("No nsamples provided for event-level explanations. Using default: 32000")
        event_dict['nsamples'] = [32000]

    if event_dict.get('tol', False):
        tolerances = event_dict.get('tol')
        if isinstance(tolerances, float):
            event_dict['tol'] = [tolerances]
        elif isinstance(tolerances, list):
            assert np.array([isinstance(x, float) for x in tolerances]).all(), "All provided tolerances must be floats."


def event_explain_all(f: Callable,
                      data: Union[List[np.ndarray], pd.DataFrame, np.array],
                      event_dict: dict,
                      pruning_data: pd.DataFrame = None,
                      baseline: Union[pd.DataFrame, np.array] = None,
                      model_features: List[Union[int, str]] = None,
                      schema: List[str] = None,
                      entity_col: Union[int, str] = None,
                      time_col: Union[int, str] = None,
                      append_to_files: bool = False,
                      verbose: bool = False,
                      max_rows_per_file: int = 1
                      ) -> pd.DataFrame:
    """Calculates event level explanations for all entities on the provided
    DataFrame applying pruning if explicit

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: Union[List[np.ndarray], pd.DataFrame, np.array]
        Sequences to be explained.
        Must contain columns with names disclosed on `model_features`.

    event_dict: dict
        Information required for the event level explanation calculation

    pruning_data: pd.DataFrame
        Pruning indexes for all sequences being explained.
        Produced by `prune_all`

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.
        The baseline can be an average event or an average sequence

    model_features: List[str]
        Features to be used by the model. Requires same order as training dataset

    schema: List[str]
        Schema of provided data

    entity_col: str
        Entity column to identify sequences

    time_col: str
        Data column that represents the time feature in order to sort sequences
        temporally

    append_to_files: bool
        Append explanations to files if file already exists

    verbose: bool
        If process is verbose
    
    max_rows_per_file: int
        Maximum number of rows per file

    Returns
    -------
    pd.DataFrame
    """
    if schema is None and isinstance(data, pd.DataFrame):
        schema = list(data.columns)
    verify_event_dict(event_dict)
    file_path = event_dict.get('path')
    make_predictions = True
    event_data = None

    tolerances_to_calc = get_tolerances_to_test(pruning_data, event_dict)

    file_index = detect_last_saved_file_index(file_path)
    num_rows_per_file = count_rows_in_last_file(file_path)
    resume_iteration = file_index * math.ceil(max_rows_per_file / num_rows_per_file)

    print(f"Resuming in file: {resume_iteration}")

    if file_path is not None and file_exists(file_path):
        event_data = read_multiple_files(file_path)
        make_predictions = False

        present_tols = set(np.unique(event_data['Tolerance'].values))
        required_tols = [x for x in tolerances_to_calc if x not in present_tols]
        if len(required_tols) == 0:
            if resume_iteration < len(data) and max_rows_per_file <= num_rows_per_file and not event_dict.get('skip_train'):
                make_predictions = True
        elif len(required_tols) == 1 and -1 in tolerances_to_calc:
            # Assuming all sequences are already explained
            make_predictions = True
        else:
            raise NotImplementedError

    if make_predictions:
        random_seeds = list(np.unique(event_dict.get('rs')))
        nsamples = list(np.unique(event_dict.get('nsamples')))
        names = ["Random Seed", "NSamples", "Event", "Shapley Value", "t (event index)", "Entity", 'Tolerance']

        if file_path is not None:
            if os.path.exists(file_path):
                assert append_to_files, "The defined path for event explanations already exists and the append option is turned off. If you wish to append the explanations please use the flag `append_to_files`, otherwise change the provided path."

        if time_col is None:
            print("No time col provided, assuming dataset is ordered ascendingly by date")

        model_features_index, entity_col_index, time_col_index = convert_to_indexes(model_features, schema, entity_col, time_col)
        data = convert_data_to_3d(data, entity_col_index, time_col_index)

        ret_event_data = []
        row_count = 0
        num_digits = 0

        for rs in random_seeds:
            for ns in nsamples:
                seq_ind = 0
                for sequence in data[resume_iteration:]:
                    if entity_col is not None:
                        entity = sequence[0, 0, entity_col_index]
                    if model_features:
                        sequence = sequence[:, :, model_features_index]
                    sequence = sequence.astype(np.float64)
                    event_data = None
                    prev_pruning_idx = None
                    for tol in tolerances_to_calc:
                        if tol == -1:
                            pruning_idx = np.ones(len(data))
                        elif pruning_data is None:
                            # we need to perform the pruning on the fly
                            pruning_idx = temp_coalition_pruning(f, sequence, baseline, tol, verbose=verbose)
                        else:
                            instance = pruning_data[pruning_data["Entity"] == entity]
                            pruning_idx = instance[instance['Tolerance'] == tol]['Pruning idx'].iloc[0]
                            pruning_idx = np.array(pruning_idx)

                            if len(pruning_idx) > sequence.shape[0]:
                                # pruning_idx reshape
                                pruning_idx = pruning_idx.reshape(len(data), -1)
                                # Use seq_ind to index into the reshaped array
                                pruning_idx = pruning_idx[seq_ind, :]

                        if np.all(prev_pruning_idx == pruning_idx):
                            # we have already calculated this, let's use it from the last iteration
                            event_data['Tolerance'] = tol
                        else:
                            local_event_dict = {'rs': rs, 'nsamples': ns}
                            event_data = local_event(f, sequence, local_event_dict, entity, entity_col, baseline, pruning_idx, verbose=verbose)
                            event_data['Event index'] = event_data['Feature'].apply(lambda x: 1 if x == 'Pruned Events' else -int(re.findall(r'\d+', x)[0])+1)
                            event_data[entity_col] = entity
                            event_data['Tolerance'] = tol

                        ret_event_data.append(event_data.values)
                        row_count += len(event_data)
                        prev_pruning_idx = np.copy(pruning_idx)

                        # Estimate number of files (assumes that all sequences have the same number of events)
                        if 0 == num_digits:
                            num_digits = min((len(data) * row_count) / max_rows_per_file, len(data))
                            num_digits = int(num_digits) + 1 if num_digits - int(num_digits) > 0 else int(num_digits)
                            
                            # Get number of digits
                            num_digits = len(str(num_digits))
                        
                        # Check if we need to write to a new file
                        if row_count >= max_rows_per_file:
                            save_multiple_files(ret_event_data, file_path, file_index, names, num_digits)
                            ret_event_data = []
                            file_index += 1
                            row_count = 0

                    seq_ind += 1

        # Save remaining data
        if ret_event_data:
            save_multiple_files(ret_event_data, file_path, file_index, names, num_digits)
        
        event_data = read_multiple_files(file_path)

        event_data['Shapley Value'] = correct_shap_vals_format(event_data)
        event_data = event_data.astype({'NSamples': 'int', 'Random Seed': 'int', 'Tolerance': 'float', 't (event index)': 'int'})

    return event_data
