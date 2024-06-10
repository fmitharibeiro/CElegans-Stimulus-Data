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
import copy
import re
import math
import altair as alt
from ...timeshap.plot.utils import multi_plot_wrapper


def plot_event_heatmap(event_data: pd.DataFrame, top_n_events: int = 20, x_multiplier: int = 1):
    """
    Plots local event explanations

    Parameters
    ----------
    event_data: pd.DataFrame
        Event global explanations
    top_n_events: int
        Number of top events to display, others will be aggregated
    x_multiplier: int
        Value to multiply the x-axis points by
    """
    # Create a deep copy of event_data
    event_data = copy.deepcopy(event_data)
    
    # Extract digit to order df by
    event_data['idx'] = event_data['Feature'].apply(
        lambda x: event_data.shape[0] if x == 'Pruned Events' else int(re.findall(r'\d+', x)[0]) - 1
    )

    # Calculate the sum of Shapley values for each event
    summed_data = event_data[event_data['Feature'] != 'Pruned Events'].groupby('Feature')['Shapley Value'].apply(
        lambda x: sum([item for sublist in x for item in sublist] if isinstance(x.iloc[0], list) else x)
    ).reset_index()
    summed_data = summed_data.sort_values('Shapley Value', ascending=False)

    # Identify top N events excluding "Pruned Events"
    top_events = summed_data.head(top_n_events)['Feature'].tolist()
    other_events = event_data[~event_data['Feature'].isin(top_events + ['Pruned Events'])]['Shapley Value'].reset_index(drop=True)

    print(f"Summed Data: {summed_data}")

    if len(other_events) > 0:
        s = other_events[0]
        for k in range(1, other_events.shape[0]):
            s = [sum(i) for i in zip(s, other_events[k])]
    else:
        s = [0] * len(event_data.iloc[0]['Shapley Value'])

    # Create 'Other Events' for the aggregated remaining events
    other_events_row = pd.DataFrame({'Feature': ['Other Events'], 'Shapley Value': [s]})

    # Combine top events, pruned events, and other events
    final_event_data = event_data[event_data['Feature'].isin(top_events + ['Pruned Events'])]
    final_event_data = pd.concat([final_event_data, other_events_row], ignore_index=True)
    # final_event_data = final_event_data.sort_values('idx')[['Shapley Value', 'Feature']]

    # Ensure Shapley Value is always a list for consistency
    final_event_data['Shapley Value'] = final_event_data['Shapley Value'].apply(lambda x: x if isinstance(x, list) else [x])
    
    # Expand the data for plotting
    expanded_data = final_event_data.explode('Shapley Value').reset_index(drop=True)
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

    # Define chart parameters
    height = 500
    width = 1000
    axis_lims = [-scale_range, scale_range]
    fontsize = 15

    # Create the chart
    c = alt.Chart().encode(
        y=alt.Y('Feature:O', axis=alt.Axis(domain=False, labelFontSize=fontsize, title=None)),
    )

    a = c.mark_rect().encode(
        x=alt.X('Output Point Multiplied:O', axis=alt.Axis(titleFontSize=fontsize, labelAngle=0)),
        color=alt.Color('rounded', title=None,
                        legend=alt.Legend(gradientLength=height,
                                          gradientThickness=10, orient='right',
                                          labelFontSize=fontsize),
                        scale=alt.Scale(domain=axis_lims, range=c_range))
    )
    b = c.mark_text(align='right', baseline='middle', dx=18, fontSize=fontsize,  # Adjust dy to push the values higher
                    color='#798184').encode(
            x=alt.X('Output Point Multiplied:O', axis=alt.Axis(orient="top", title='Shapley Value', domain=False,
                                                              titleY=height + 20, titleX=172, labelAngle=0,
                                                              labelFontSize=fontsize)),
            text='rounded_str',
    )

    event_plot = alt.layer(a, b, data=expanded_data).properties(
        width=math.ceil(0.8 * width),
        height=height
    )

    return event_plot


def plot_global_event(event_data: pd.DataFrame,
                      plot_parameters: dict = None,
                      ):
    """Plots global event explanations

    Parameters
    ----------
    event_data: pd.DataFrame
        Event global explanations

    plot_parameters: dict
        Dict containing optional plot parameters
            'height': height of the plot, default 150
            'width': width of the plot, default 360
            'axis_lims': plot Y domain, default [-0.3, 0.9]
            't_limit': number of events to plot, default -20
    """
    def plot(event_data: pd.DataFrame, plot_parameters: dict = None):
        event_data = copy.deepcopy(event_data)
        event_data = event_data[event_data['t (event index)'] < 1]
        event_data = event_data[['Shapley Value', 't (event index)']]

        # Related to issue #43; credit to @edpclau
        event_data = copy.deepcopy(event_data)
        avg_df = event_data.groupby('t (event index)').mean().reset_index()
        event_data['type'] = 'Shapley Value'
        avg_df['type'] = 'Mean'
        event_data = pd.concat([event_data, avg_df], axis=0, ignore_index=True)

        if plot_parameters is None:
            plot_parameters = {}

        height = plot_parameters.get('height', 150)
        width = plot_parameters.get('width', 360)
        axis_lims = plot_parameters.get('axis_lim', [-0.3, 0.9])
        t_limit = plot_parameters.get('axis_lim', -20)

        event_data = event_data[event_data['t (event index)'] >= t_limit]
        event_data = event_data[event_data['Shapley Value'] >= axis_lims[0]]
        event_data = event_data[event_data['Shapley Value'] <= axis_lims[1]]

        global_event = alt.Chart(event_data).mark_point(stroke='white',
                                                      strokeWidth=.6).encode(
            y=alt.Y('Shapley Value', axis=alt.Axis(grid=True, titleX=-23),
                    title="Shapley Value", scale=alt.Scale(domain=axis_lims, )),
            x=alt.X('t (event index):O', axis=alt.Axis(labelAngle=0)),
            color=alt.Color('type',
                            scale=alt.Scale(domain=['Shapley Value', 'Mean'],
                                            range=["#48caaa", '#d76d58']),
                            legend=alt.Legend(title=None, fillColor="white",
                                              symbolStrokeWidth=0, symbolSize=50,
                                              orient="top-left")),
            opacity=alt.condition(alt.datum.type == 'Mean', alt.value(1.0),
                                  alt.value(0.2)),
            size=alt.condition(alt.datum.type == 'Mean', alt.value(70),
                               alt.value(30)),
        ).properties(
            width=width,
            height=height,
        )

        return global_event

    return multi_plot_wrapper(event_data, plot, ((plot_parameters),))
