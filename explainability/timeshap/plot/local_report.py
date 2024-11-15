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
import pandas as pd
from ...timeshap.plot import plot_temp_coalition_pruning, plot_event_heatmap, plot_feat_barplot, plot_cell_level, plot_feat_heatmap
from ...timeshap.explainer import prune_given_data
from ...timeshap.explainer.extra import correct_shap_vals_format, max_abs_value


def plot_local_report(pruning_dict: dict,
                      event_dict: dict,
                      feature_dict: dict,
                      cell_dict: dict,
                      coal_plot_data: pd.DataFrame = None,
                      event_data: pd.DataFrame = None,
                      feat_data: pd.DataFrame = None,
                      cell_data: pd.DataFrame = None,
                      ):
    """Plots a local report given explanations

    Parameters
    ----------
    pruning_dict: dict
        Information required for the pruning algorithm

    event_dict: dict
        Information required for the event level explanation calculation

    feature_dict: dict
        Information required for the feature level explanation calculation

    cell_dict: dict
        Information required for the cell level explanation calculation

    coal_plot_data: pd.DataFrame
        Pruning algorithm data to plot

    event_data: pd.DataFrame
        Event explanations to plot

    feat_data: pd.DataFrame
        Feature explanations to plot

    cell_data: pd.DataFrame
        Cell explanations to plot
    """
    if pruning_dict is None:
        if pruning_dict is not None:
            assert pruning_dict.get('path', False), "No data or path to data provided to calculate pruning statistics"
    if event_data is None:
        assert event_dict.get('path', False), "No data or path to data provided to plot event explanations"
    if feat_data is None:
        assert feature_dict.get('path', False), "No data or path to data provided to plot feature explanations"
    if cell_data is None and cell_dict is not None:
        assert cell_dict.get('path', False), "No data or path to data provided to plot feature explanations"

    if coal_plot_data is None:
        if pruning_dict is not None and pruning_dict.get('path'):
            coal_plot_data = pd.read_csv(pruning_dict.get('path'))
            coal_plot_data['Shapley Value'] = correct_shap_vals_format(coal_plot_data)
    if event_data is None:
        event_data = pd.read_csv(event_dict.get('path'))
    if feat_data is None:
        feat_data = pd.read_csv(feature_dict.get('path'))
    if cell_data is None and cell_dict is not None:
        cell_data = pd.read_csv(cell_dict.get('path'))

    num_pts = 100
    f = lambda x: [x[i] for i in range(0, len(x), int(len(x)/num_pts))]
    if coal_plot_data is not None:
        coal_prun_idx = prune_given_data(coal_plot_data, pruning_dict.get('tol'))
        plot_lim = len(coal_prun_idx)
        coal_plot_data['Shapley Value'] = correct_shap_vals_format(coal_plot_data)
        if isinstance(coal_plot_data['Shapley Value'], list):
            coal_plot_data['Shapley Value'] = coal_plot_data['Shapley Value'].apply(lambda x: sum([abs(a) for a in x])/len(x))
        pruning_plot = plot_temp_coalition_pruning(coal_plot_data, coal_prun_idx, plot_lim)

    event_data['Shapley Value'] = correct_shap_vals_format(event_data)
    l = len(event_data['Shapley Value'][0])
    event_data['Shapley Value'] = event_data['Shapley Value'].apply(f)
    event_plot = plot_event_heatmap(event_data, x_multiplier=int(l/num_pts))

    feat_data['Shapley Value'] = correct_shap_vals_format(feat_data)
    feat_data['Shapley Value'] = feat_data['Shapley Value'].apply(f)
    feature_plot = plot_feat_heatmap(feat_data, x_multiplier=int(l/num_pts))

    if cell_dict:
        cell_data['Shapley Value'] = correct_shap_vals_format(cell_data)
        cell_data['Shapley Value'] = cell_data['Shapley Value'].apply(f)

        feat_names = list(feat_data['Feature'].values)[:-1]  # exclude pruned events
        cell_plot = plot_cell_level(cell_data, feat_names, x_multiplier=int(l/num_pts))
        if coal_plot_data is not None:
            plot_report = (pruning_plot | event_plot | feature_plot | cell_plot).resolve_scale(color='independent')
        else:
            plot_report = (event_plot | feature_plot | cell_plot).resolve_scale(color='independent')

    else:
        if coal_plot_data is not None:
            plot_report = (pruning_plot | event_plot | feature_plot).resolve_scale( color='independent')
        else:
            plot_report = (event_plot | feature_plot).resolve_scale( color='independent')

    return plot_report
