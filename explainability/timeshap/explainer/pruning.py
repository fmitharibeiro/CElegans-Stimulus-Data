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

from typing import Callable, Union, Tuple, List
import numpy as np
import pandas as pd
from ...timeshap.explainer.kernel import TimeShapKernel
import os
from pathlib import Path
from tqdm import tqdm
from ...timeshap.utils import convert_to_indexes, convert_data_to_3d
from ...timeshap.explainer.extra import correct_shap_vals_format, plot_pruning_data, save_multiple_files, read_multiple_files, file_exists


def calc_prun_indexes(df: pd.DataFrame,
                      tol: Union[float, int, list]
                      ) -> pd.DataFrame:
    """Calculates the pruning indexes given pruning data and tolerances

    Parameters
    ----------
    df: pd.DataFrame
        Pruning data to be analysed produced by `prune_all`

    tol: Union[float, list]
        The tolerances to analyze the pruning

    Returns
    -------
    pd.DataFrame
    """
    if "Tolerance" not in list(df.columns):
        pruning_data = []
        if not isinstance(tol, list):
            tol = [tol]
        for uuid in np.unique(df.iloc[:, -1].values):
            uuid_data = df[df.iloc[:, -1] == uuid]
            pruning_data.append([uuid, -1, -(uuid_data.shape[0] / 2) + 1])
            for tolerance in tol:
                pruning_idx = prune_given_data(uuid_data, tolerance)
                pruning_data.append([uuid, tolerance, pruning_idx])

        df = pd.DataFrame(pruning_data, columns=["Entity", 'Tolerance', 'Pruning idx'])
    return df


def pruning_statistics(df: pd.DataFrame,
                       tol: Union[float, list],
                       ) -> pd.DataFrame:
    """Calculates global pruning statistics with the given tolerances.

    Parameters
    ----------
    df: pd.DataFrame
        Pruning data to be analysed produced by `prune_all`

    tol: Union[float, list]
        The tolerances to analyze the pruning

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(tol, float):
        tol = [tol]
    if "Tolerance" not in list(df.columns):
        df = calc_prun_indexes(df, tol)

    resume = []
    orig = df[df['Tolerance'] == -1]
    for idx, row in orig.iterrows():
        resume += [["Original", 'No Pruning', row["Entity"], -row['Pruning idx']]]

    for tol in tol:
        tolerance_sequences = df[df['Tolerance'] == tol]
        for idx, row in tolerance_sequences.iterrows():
            resume.append(["Pruning", tol,  row["Entity"], -row['Pruning idx']])

    resume_df = pd.DataFrame(resume, columns=["Algorithm", "Tolerance", "Entity", "Sequence Length"])
    resume_df['Mean'] = resume_df['Sequence Length']
    resume_df['Std'] = resume_df['Sequence Length']
    resume_df = resume_df.groupby("Tolerance").agg({"Mean": "mean", "Std": "std"})
    resume_df.reset_index(inplace=True)
    resume_df = resume_df.rename(columns={'index': 'Tolerance'})
    return resume_df


def prune_given_data(data: pd.DataFrame,
                     tolerance: float,
                     ) -> int:
    """Calculates the pruning index to prune the sequence to with a given tolerance

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe containing the pruning algorithm information

    tolerance: str
        Tolerance to prun the sequence

    Returns
    -------
    int
    """
    data = data[data['Coalition'] == 'Sum of contribution of events \u2264 t']
    if tolerance == 0:
        # to filter float unprecision out
        tolerance = 0.00000000001
    data = data.iloc[::-1].reset_index(drop=True)
    # print(f"data pruning: {data['Pruning']}")
    
    # Prune if all elements in 'Pruning' are 1
    if (data['Pruning'] == 1).all():
        # data['Shapley Value'] = correct_shap_vals_format(data)
        # # Convert 'Shapley Value' to the mean of absolute values
        # data['Mean Shapley Value'] = data['Shapley Value'].apply(lambda x: np.mean(np.abs(x)))

        # Create a copy of 'Pruning' to modify
        pruned_list = data['Pruning'].copy()

        # Check if the mean of absolute 'Shapley Value' is below or equal to tolerance
        # pruned_list[data['Mean Shapley Value'] <= tolerance] = 0
        pruned_list[data['Shapley Value'] <= tolerance] = 0

        # Prune consecutive close values
        for i in range(1, len(data)):
            # if np.abs(data['Mean Shapley Value'].iloc[i] - data['Mean Shapley Value'].iloc[i-1]) <= 0.00001:
            if np.abs(data['Shapley Value'].iloc[i] - data['Shapley Value'].iloc[i-1]) <= 0.00001:
                pruned_list.iloc[i] = 0

        # If the second value is pruned, prune the first value as well
        if pruned_list.iloc[1] == 0:
            pruned_list.iloc[0] = 0

        # Update 'Pruning' with the pruned list
        data['Pruning'] = pruned_list

    return data['Pruning'] # Returns the list
    # respecting_lens = data[data['Shapley Value'].abs() <= tolerance]
    # if respecting_lens.shape[0] == 0:
    #     return -data['t (event index)'].min()
    # return respecting_lens.iloc[0]['t (event index)']


def temp_coalition_pruning(f: Callable,
                           data: np.ndarray,
                           baseline: Union[np.ndarray, pd.DataFrame],
                           tolerance: float = None,
                           ret_plot_data=False,
                           verbose=False,
                           ) -> Union[int, pd.DataFrame, Tuple[int, pd.DataFrame]]:
    """Temporal coalition pruning method

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]im
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: numpy.ndarray
        Input matrix to use. First element of the first dimension is explained,
        using the rest of the elements as context/hidden state.

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.
        The baseline can be an average event or an average sequence

    tolerance: float
        Temporal coalition explainer tolerance.
        Represents the maximum allowed Shapley Value of the older grouped events.

    ret_plot_data: bool
        If method returns pruning algorithm across the whole sequence

    verbose: bool
        If process is verbose

    Returns
    -------
    Union[int, pd.DataFrame, Tuple[int, pd.DataFrame]]:
        int:
            Pruning index
        pd.DataFrame
            Pruning data over the whole sequence
         Tuple[int, pd.DataFrame]]
            Pruning index and Pruning data over the whole sequence
    """
    if verbose:
        print("Allowed importance for pruned events: {}".format(tolerance))

    if ret_plot_data:
        plot_pruning_out = [0] * data.shape[1]
        plot_pruning_in = [0] * data.shape[1]
        plot_data = []
    pruning_idx = np.ones(data.shape[1], dtype=int) # All indexes start inside the group
    prev_value = 0
    first_iteration = True
    for seq_len in tqdm(range(data.shape[1]-1, -1, -1), desc='Pruning Events') if not verbose else range(data.shape[1]-1, -1, -1):
        explainer = TimeShapKernel(f, baseline, 0, "pruning")
        shap_values = explainer.shap_values(data, pruning_idx=seq_len, **{'nsamples': 4, 'verbose': verbose})

        if first_iteration and tolerance:
            tolerance = tolerance * np.mean(abs(shap_values[1]))
            first_iteration = False

        if verbose:
            print("len {} | importance {} | sign {} | pruned? {}".format(-data.shape[1] + seq_len,
                        np.mean(abs(shap_values[1])),
                        np.sign(shap_values[1][np.argmax(abs(shap_values[1]))]),
                        "Yes" if tolerance and (np.mean(abs(shap_values[1])) <= tolerance or np.isclose(np.mean(abs(shap_values[1])), prev_value, atol=0.00001)) else "No"))

        if tolerance and seq_len == data.shape[1] and np.mean(abs(shap_values[1])) <= tolerance:
            print("Unable to prune sequence.")
        
        if tolerance and np.isclose(np.mean(abs(shap_values[1])), prev_value, atol=0.00001):
            pruning_idx[-data.shape[1] + seq_len] = 0

            if seq_len == data.shape[1]-2: # Remove first event if the second one is removed
                pruning_idx[-1] = 0

        if seq_len < data.shape[1] and tolerance and np.mean(abs(shap_values[1])) <= tolerance:
            if np.all(pruning_idx[:-data.shape[1] + seq_len + 1] == 1):
                pruning_idx[:-data.shape[1] + seq_len + 1] = 0
            if not ret_plot_data:
                return pruning_idx
            
        if ret_plot_data:
            plot_pruning_out[-data.shape[1]+seq_len] = np.mean(abs(shap_values[0]))
            plot_pruning_in[-data.shape[1]+seq_len] = np.mean(abs(shap_values[1]))
            plot_data += [['Sum of contribution of events \u003E t', -data.shape[1]+seq_len, pruning_idx[-data.shape[1] + seq_len], np.mean(abs(shap_values[0]))]]
            plot_data += [['Sum of contribution of events \u2264 t', -data.shape[1]+seq_len, pruning_idx[-data.shape[1] + seq_len], np.mean(abs(shap_values[1]))]]
        
        prev_value = np.mean(abs(shap_values[1]))

    # if tolerance is not None and pruning_idx == 0:
    #     pruning_idx = -data.shape[1]

    if tolerance is not None and ret_plot_data:
        # used for plotting
        return pruning_idx,  pd.DataFrame(plot_data, columns=['Coalition', 't (event index)', 'Pruning', 'Shapley Value']), (tolerance, plot_pruning_out, plot_pruning_in)
    if tolerance is not None and not ret_plot_data:
        # used for event level
        return pruning_idx
    return pd.DataFrame(plot_data, columns=['Coalition', 't (event index)', 'Pruning', 'Shapley Value'])


def local_pruning(f: Callable[[np.ndarray], np.ndarray],
                  data: np.ndarray,
                  pruning_dict: dict,
                  baseline: Union[np.ndarray, pd.DataFrame],
                  entity_uuid: Union[str, int, float] = None,
                  entity_col: str = None,
                  verbose: bool = False,
                  ) -> Tuple[pd.DataFrame, int]:
    """Method to prune a sequence or fetch the respective information if a path
    is provided

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: numpy.ndarray
        Input matrix to use. First element of the first dimension is explained,
        using the rest of the elements as context/hidden state.

    pruning_dict: dict
        Information required for pruning algorithm

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.
        The baseline can be an average event or an average sequence

    entity_uuid: Union[str, int, float]
        The indentifier of the sequence that is being pruned.
        Used when fetching information from a csv of explanations

    entity_col: str
        Column that contains the sequence identifiers
        Used when fetching information from a csv of explanations

    verbose: bool
        If process is verbose

    Returns
    -------
    Tuple[int, pd.DataFrame]]
            Pruning index and Pruning data over the whole sequence
    """
    def calculate_pruning():
        if baseline is None:
            raise ValueError("Baseline is not defined")
        coal_prun_idx, coal_plot_data, coal_plot = temp_coalition_pruning(f,
                                                               data,
                                                               baseline,
                                                               pruning_dict['tol'],
                                                               ret_plot_data=True,
                                                               verbose=verbose)

        return coal_prun_idx, coal_plot_data, coal_plot

    if pruning_dict.get("path") is None or not os.path.exists(pruning_dict.get("path")):
        #print("No path to explainer data provided. Calculating data")
        if baseline is None:
            raise ValueError("Baseline is not defined")
        coal_prun_idx, coal_plot_data, coal_plot = calculate_pruning()
        if pruning_dict.get("path") is not None:
            # create directory
            if '/' in pruning_dict.get("path"):
                Path(pruning_dict.get("path").rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
            coal_plot_data.to_csv(pruning_dict.get("path"), index=False)
            plot_pruning_data(coal_plot[0], coal_plot[1], coal_plot[2], pruning_dict.get("path").rsplit(".", 1)[0]+".png")

    elif pruning_dict.get("path") is not None and os.path.exists(pruning_dict.get("path")):
        # TODO
        coal_plot_data = pd.read_csv(pruning_dict.get("path"))
        if len(coal_plot_data.columns) > 4:
            # global df
            assert entity_uuid is not None, "When using a dataset with several instances, a uuid needs to be provided"
            coal_plot_data = coal_plot_data[coal_plot_data[entity_col] == entity_uuid]
        coal_prun_idx = prune_given_data(coal_plot_data, pruning_dict.get('tol'))
    else:
        raise ValueError('Unrecognized explainer procedure.')
    return coal_plot_data, coal_prun_idx


def verify_pruning_dict(pruning_dict: dict):
    """Verifies the format of the pruning dict

    Parameters
    ----------
    pruning_dict: dict
    """
    if pruning_dict.get('path'):
        assert isinstance(pruning_dict.get('path'), str)

    assert pruning_dict.get('tol', False), "Tolerance(s) must be provided on the pruning dict"
    tolerances = pruning_dict.get('tol')
    if isinstance(tolerances, float):
        pruning_dict['tol'] = [tolerances]
    elif isinstance(tolerances, list):
        assert np.array([isinstance(x, float) for x in tolerances]).all(), "All provided tolerances must be floats."
    else:
        raise ValueError("Unsuported format of pruning tolerance(s). Please provide one tolerance or a list of them.")


def prune_all(f: Callable,
              data: Union[List[np.ndarray], pd.DataFrame, np.array],
              pruning_dict: dict,
              baseline: Union[pd.DataFrame, np.array] = None,
              model_features: List[Union[int, str]] = None,
              schema: List[str] = None,
              entity_col: Union[int, str] = None,
              time_col: Union[int, str] = None,
              append_to_files: bool = False,
              verbose: bool = False,
              max_rows_per_file: int = 4000  # New parameter to control file size
              ) -> pd.DataFrame:
    """Applies pruning to a dataset

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: Union[pd.DataFrame, np.array]
        Sequence to explain.

    pruning_dict: dict
        Information required for pruning algorithm

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.
        The baseline can be an average event or an average sequence

    model_features: List[str]
        In-order list of features to select and input to the model

    schema: List[str]
        Schema of provided data

    entity_col: str
        Column that contains the sequence identifiers

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
    verify_pruning_dict(pruning_dict)
    file_path = pruning_dict.get('path')
    tolerances = list(np.unique(pruning_dict.get('tol')))
    make_predictions = True
    prun_data = None
    if file_path is not None and file_exists(file_path):
        prun_data = read_multiple_files(file_path)
        make_predictions = False

        # TODO resume explanations for missing entities
        # necessary_entities = set(np.unique(data[entity_col].values))
        # loaded_csv = pd.read_csv(file_path)
        # present_entities = set(np.unique(loaded_csv[entity_col].values))
        # if necessary_entities.issubset(present_entities):
        #     make_predictions = False
        #     prun_data = loaded_csv[loaded_csv[entity_col].isin(necessary_entities)]

    if make_predictions:
        ret_prun_data = []
        file_index = 0
        row_count = 0
        num_digits = 0

        if entity_col:
            names = ["Coalition", "t (event index)", "Pruning", "Shapley Value", entity_col if isinstance(entity_col, str) else "Entity"]
        else:
            names = ['Coalition', 't (event index)', 'Pruning', 'Shapley Value']
        
        if time_col is None:
            print("No time col provided, assuming dataset is ordered ascendingly by date")

        model_features_index, entity_col_index, time_col_index = convert_to_indexes(model_features, schema, entity_col, time_col)
        data = convert_data_to_3d(data, entity_col_index, time_col_index)
        
        for sequence in data:
            if entity_col is not None:
                entity = sequence[0, 0, entity_col_index]
            if model_features:
                sequence = sequence[:, :, model_features_index]
            sequence = sequence.astype(np.float64)
            local_pruning_data = temp_coalition_pruning(f, sequence, baseline, None, ret_plot_data=True, verbose=verbose)

            if entity_col is not None:
                local_pruning_data["Entity"] = entity

            ret_prun_data.append(local_pruning_data.values)
            row_count += len(local_pruning_data)

            # Estimate number of files (assumes that all sequences have the same number of events)
            if 0 == num_digits:
                num_digits = (len(data) * row_count) / max_rows_per_file
                num_digits = int(num_digits) + 1 if num_digits - int(num_digits) > 0 else int(num_digits)
                
                # Get number of digits
                num_digits = len(str(num_digits))
            
            # Check if we need to write to a new file
            if row_count >= max_rows_per_file:
                save_multiple_files(ret_prun_data, file_path, file_index, names, num_digits)
                ret_prun_data = []
                file_index += 1
                row_count = 0

        # Save remaining data
        if ret_prun_data:
            save_multiple_files(ret_prun_data, file_path, file_index, names, num_digits)
        
        prun_data = read_multiple_files(file_path)

    df = calc_prun_indexes(prun_data, tolerances)
    return df
