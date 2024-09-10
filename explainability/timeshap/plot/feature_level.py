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
import numpy as np
import copy
import altair as alt
import math
from ...timeshap.plot.utils import multi_plot_wrapper
from ...timeshap.explainer.extra import correct_shap_vals_format


def plot_feat_barplot(feat_data: pd.DataFrame,
                      top_x_feats: int = 15,
                      plot_features: dict = None
                      ):
    """Plots local feature explanations

    Parameters
    ----------
    feat_data: pd.DataFrame
        Feature explanations

    top_x_feats: int
        The number of feature to display.

    plot_features: dict
        Dict containing mapping between model features and display features
    """
    feat_data = copy.deepcopy(feat_data)
    if plot_features:
        plot_features['Pruned Events'] = 'Pruned Events'
        feat_data['Feature'] = feat_data['Feature'].apply(lambda x: plot_features[x])

    feat_data['sort_col'] = feat_data['Shapley Value'].apply(lambda x: abs(x))

    if top_x_feats is not None and feat_data.shape[0] > top_x_feats:
        sorted_df = feat_data.sort_values('sort_col', ascending=False)
        cutoff_contribution = abs(sorted_df.iloc[4]['Shapley Value'])
        feat_data = feat_data[np.logical_or(feat_data['Explanation'] >= cutoff_contribution, feat_data['Explanation'] <= -cutoff_contribution)]
    
    min_shapley_value = feat_data['Shapley Value'].min()
    max_shapley_value = feat_data['Shapley Value'].max()

    axis_lims = [min(0, min_shapley_value), max(0, max_shapley_value)]

    a = alt.Chart(feat_data).mark_bar(size=15, thickness=1).encode(
        y=alt.Y("Feature", axis=alt.Axis(title="Feature", labelFontSize=15,
                                         titleFontSize=15, titleX=-61),
                sort=alt.SortField(field='sort_col', order='descending')),
        x=alt.X('Shapley Value', axis=alt.Axis(grid=True, title="Shapley Value",
                                            labelFontSize=15, titleFontSize=15),
                scale=alt.Scale(domain=axis_lims)),
    )

    line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(
        color='#798184').encode(x='x')

    feature_plot = (a + line).properties(
        width=190,
        height=225
    )
    return feature_plot


def plot_feat_heatmap(feat_data: pd.DataFrame, top_x_feats: int = 15, plot_features: dict = None, x_multiplier: int = 1):
    """
    Plots local feature explanations

    Parameters
    ----------
    feat_data: pd.DataFrame
        Feature explanations
    top_x_feats: int
        The number of features to display.
    plot_features: dict
        Dict containing mapping between model features and display features
    x_multiplier: int
        Value to multiply the x-axis points by
    """
    # Create a deep copy of feat_data
    feat_data = copy.deepcopy(feat_data)

    # Apply feature mapping if provided
    if plot_features:
        feat_data['Feature'] = feat_data['Feature'].apply(lambda x: plot_features.get(x, x))

    # Calculate the max of the absolute value of Shapley values for each feature
    summed_data = feat_data.groupby('Feature')['Shapley Value'].apply(
        lambda x: max([abs(item) for sublist in x for item in sublist] if isinstance(x.iloc[0], list) else max(abs(x)))
    ).reset_index()
    summed_data = summed_data.sort_values('Shapley Value', ascending=False)

    # Identify top X features
    top_feats = summed_data.head(top_x_feats)['Feature'].tolist()
    final_feat_data = feat_data[feat_data['Feature'].isin(top_feats)]

    # Ensure Shapley Value is always a list for consistency
    final_feat_data['Shapley Value'] = final_feat_data['Shapley Value'].apply(lambda x: x if isinstance(x, list) else [x])
    
    # Expand the data for plotting
    expanded_data = final_feat_data.explode('Shapley Value').reset_index(drop=True)
    expanded_data['Output Point'] = expanded_data.groupby('Feature').cumcount()
    expanded_data['Output Point Multiplied'] = expanded_data['Output Point'] * x_multiplier

    # Prepare for plotting
    c_range = ["#5f8fd6", "#99c3fb", "#f5f5f5", "#ffaa92", "#d16f5b"]

    expanded_data['rounded'] = expanded_data['Shapley Value'].apply(lambda x: round(x, 3))
    expanded_data['rounded_str'] = expanded_data['Shapley Value'].apply(
        lambda x: '___' if round(x, 3) == 0 else str(round(x, 3))
    )
    expanded_data['rounded_str'] = expanded_data['rounded_str'].apply(
        lambda x: f'{x}0' if len(x) == 4 else x
    )

    min_shapley_value = expanded_data['Shapley Value'].min()
    max_shapley_value = expanded_data['Shapley Value'].max()
    scale_range = max(abs(min_shapley_value), abs(max_shapley_value))

    # Clip points at the beginning and the end where all values are '___'
    grouped_data = expanded_data.groupby('Output Point Multiplied').apply(
        lambda g: (g['rounded'] == 0.0).all()
    )

    def trim_edges_to_single_false_group(grouped_data):
        first_false_idx = grouped_data.idxmin()  # First occurrence of False
        last_false_idx = len(grouped_data) - grouped_data[::-1].idxmin() - 1  # Last occurrence of False
        for i in range(first_false_idx, last_false_idx + 1):
            grouped_data[i] = False
        return grouped_data

    # Ensure plot data is contiguous
    grouped_data = trim_edges_to_single_false_group(grouped_data)

    clipped_data = expanded_data[~expanded_data['Output Point Multiplied'].isin(grouped_data[grouped_data].index)]
    clipped_pts = len(grouped_data[grouped_data])

    # Define chart parameters
    height = 500
    # width = (100000 // x_multiplier) - clipped_pts * len(grouped_data)
    width = 50*(len(grouped_data)-len(grouped_data[grouped_data]))
    axis_lims = [-scale_range, scale_range]
    fontsize = 15

    # Create the chart
    c = alt.Chart().encode(
        y=alt.Y('Feature:O', axis=alt.Axis(domain=False, labelFontSize=fontsize, title=None)),
    )

    a = c.mark_rect().encode(
        x=alt.X('Output Point Multiplied:O', axis=alt.Axis(titleFontSize=fontsize, labelAngle=0, title='Shapley Values of Features VS Output Points', titleX=width / 2)),
        color=alt.Color('rounded', title=None,
                        legend=alt.Legend(gradientLength=height,
                                          gradientThickness=10, orient='right',
                                          labelFontSize=fontsize),
                        scale=alt.Scale(domain=axis_lims, range=c_range))
    )

    b = c.mark_text(align='center', baseline='middle', dy=0, fontSize=fontsize,  # Adjust dy to move the text up
                    color='#798184').encode(
        x=alt.X('Output Point Multiplied:O'),
        text='rounded_str',
    )

    feature_plot = alt.layer(a, b, data=clipped_data).properties(
        width=math.ceil(0.8 * width),
        height=height
    )

    return feature_plot


def plot_global_feat(feat_data: pd.DataFrame,
                     top_x_feats: int = 12,
                     threshold: float = None,
                     plot_features: dict = None,
                     plot_parameters: dict = None,
                     **kwargs
                     ):
    """ Plots global feature plots

    Parameters
    ----------
    feat_data: pd.DataFrame
        Feature explanations to plot

    top_x_feats: int
        The number of feature to display.

    threshold: float
        The minimum absolute importance that a feature needs to have to be displayed

    plot_features: dict
        Dict containing mapping between model features and display features

    plot_parameters: dict
        Dict containing optional plot parameters
            'height': height of the plot, default 280
            'width': width of the plot, default 288
            'axis_lims': plot Y domain, default [-0.2, 0.6]
            'FontSize': plot font size, default 13
    """
    def plot(feat_data, top_x_feats, threshold, plot_features, plot_parameters):
        # Correct the Shapley Values format
        feat_data['Shapley Value'] = correct_shap_vals_format(feat_data)

        # TODO: Correct
        feat_data['Shapley Value'] = feat_data['Shapley Value'].apply(lambda x: x[500])

        avg_df = feat_data.groupby('Feature').mean()['Shapley Value']
        if threshold is None and len(avg_df) >= top_x_feats:
            sorted_series = avg_df.abs().sort_values(ascending=False)
            threshold = sorted_series.iloc[top_x_feats-1]
        if threshold:
            avg_df = avg_df[np.logical_or(avg_df <= -threshold, avg_df >= threshold)]
        feat_data = feat_data[feat_data['Feature'].isin(avg_df.index)][['Shapley Value', 'Feature']]

        if threshold:
            # Related to issue #43; credit to @edpclau
            avg_df = pd.concat([avg_df, pd.Series([0], index=['(...)'])],axis=0)
            feat_data = pd.concat([feat_data,
                                   pd.DataFrame({'Feature': '(...)',
                                                 'Shapley Value': -0.6, },
                                                index=[0])], ignore_index=True, axis=0)

        feat_data['type'] = 'Shapley Value'

        for index, value in avg_df.items():
            if index == '(...)':
                # Related to issue #43; credit to @edpclau
                feat_data = pd.concat([feat_data,
                                       pd.DataFrame({'Feature': index,
                                                     'Shapley Value': None,
                                                     'type': 'Mean'},
                                                    index=[0])],
                                      ignore_index=True,
                                      axis=0)
            else:
                # Related to issue #43; credit to @edpclau
                feat_data = pd.concat([feat_data,
                                       pd.DataFrame({'Feature': index,
                                                     'Shapley Value': value,
                                                     'type': 'Mean'},
                                                    index=[0])],
                                      ignore_index=True,
                                      axis=0)

        sort_features = list(avg_df.sort_values(ascending=False).index)
        if plot_features:
            plot_features = copy.deepcopy(plot_features)
            plot_features['Pruned Events'] = 'Pruned Events'
            plot_features['(...)'] = '(...)'
            feat_data['Feature'] = feat_data['Feature'].apply(lambda x: plot_features[x])
            sort_features = [plot_features[x] for x in sort_features]

        if plot_parameters is None:
            plot_parameters = {}
        height = plot_parameters.get('height', 280)
        width = plot_parameters.get('width', 400)
        axis_lims = plot_parameters.get('axis_lim', [min(feat_data['Shapley Value']), max(feat_data['Shapley Value'])])
        fontsize = plot_parameters.get('FontSize', 13)

        global_feats_plot = alt.Chart(feat_data).mark_point(stroke='white',
                                                             strokeWidth=.6).encode(
            x=alt.X('Shapley Value', axis=alt.Axis(title='Shapley Value', grid=True),
                    scale=alt.Scale(domain=axis_lims)),
            y=alt.Y('Feature:O',
                    sort=sort_features,
                    axis=alt.Axis(labelFontSize=fontsize, titleX=-51)),
            color=alt.Color('type',
                            scale=alt.Scale(domain=['Shapley Value', 'Mean'],
                                            range=["#618FE0", '#d76d58']),
                            legend=alt.Legend(title=None, fillColor="white",
                                              symbolStrokeWidth=0, symbolSize=50,
                                              orient="bottom-right")),
            opacity=alt.condition(alt.datum.type == 'Mean', alt.value(1.0),
                                  alt.value(0.1)),
            size=alt.condition(alt.datum.type == 'Mean', alt.value(70),
                               alt.value(30)),
        ).properties(
            width=width,
            height=height
        )
        return global_feats_plot

    return multi_plot_wrapper(feat_data, plot, (top_x_feats, threshold, plot_features, plot_parameters))
