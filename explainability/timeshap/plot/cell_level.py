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
import altair as alt
import re
import copy
from typing import List
import math


import altair as alt
import pandas as pd
import numpy as np
import re
import copy
import math
from typing import Optional

def plot_cell_level(cell_data: pd.DataFrame, 
                    top_n_events: int = 30, 
                    x_multiplier: int = 1, 
                    plot_parameters: Optional[dict] = None):
    """
    Plots local feature explanations for a single feature against events.

    Parameters
    ----------
    cell_data: pd.DataFrame
        Cell level explanations

    top_n_events: int
        Number of top events to display, others will be aggregated

    x_multiplier: int
        Value to multiply the x-axis points by

    plot_parameters: dict, optional
        Dictionary containing optional plot parameters:
            'height': height of the plot, default 225
            'width': width of the plot, default 1000
            'axis_lims': plot Y domain, default [-0.5, 0.5]
            'FontSize': plot font size, default 15
    """
    # Filter cell_data to the selected feature
    cell_data = copy.deepcopy(cell_data)
    cell_data = cell_data[cell_data['Feature'] == 'Feature 1']

    # Rank events based on Shapley Value to find top N events
    event_ranking = cell_data[cell_data['Event'] != 'Pruned Events'].groupby('Event')['Shapley Value'].apply(
        lambda x: max([abs(item) for sublist in x for item in sublist] if isinstance(x.iloc[0], list) else max(abs(x)))
    ).reset_index()
    event_ranking = event_ranking.sort_values('Shapley Value', ascending=False)

    # Select top events and aggregate others
    top_events = event_ranking.head(top_n_events)['Event'].tolist()
    other_events = cell_data[~cell_data['Event'].isin(top_events + ['Pruned Events'])]['Shapley Value'].reset_index(drop=True)

    # Aggregate other events into "Other Events"
    if not other_events.empty:
        other_value = max(other_events, key=abs)
        other_events_row = pd.DataFrame({'Event': ['Other Events'], 'Shapley Value': [other_value]})
    else:
        other_events_row = pd.DataFrame(columns=['Event', 'Shapley Value'])

    # Combine top events, pruned events, and "Other Events"
    filtered_data = cell_data[cell_data['Event'].isin(top_events + ['Pruned Events'])]
    filtered_data = pd.concat([filtered_data, other_events_row], ignore_index=True)

    # Ensure Shapley Value is always a list
    filtered_data['Shapley Value'] = filtered_data['Shapley Value'].apply(lambda x: x if isinstance(x, list) else [x])

    # Expand data for plotting
    expanded_data = filtered_data.explode('Shapley Value').reset_index(drop=True)
    expanded_data['Output Point'] = expanded_data.groupby('Event').cumcount()
    expanded_data['Output Point Multiplied'] = expanded_data['Output Point'] * x_multiplier

    # Prepare color range and other settings
    c_range = ["#5f8fd6", "#99c3fb", "#f5f5f5", "#ffaa92", "#d16f5b"]

    expanded_data['rounded'] = expanded_data['Shapley Value'].apply(lambda x: round(x, 3))
    expanded_data['rounded_str'] = expanded_data['Shapley Value'].apply(
        lambda x: '___' if round(x, 3) == 0 else str(round(x, 3))
    )

    # Set axis limits and font size
    min_shapley_value = expanded_data['Shapley Value'].min()
    max_shapley_value = expanded_data['Shapley Value'].max()
    scale_range = max(abs(min_shapley_value), abs(max_shapley_value))

    if plot_parameters is None:
        plot_parameters = {}
    height = plot_parameters.get('height', 225)
    width = plot_parameters.get('width', 1000)
    axis_lims = plot_parameters.get('axis_lims', [-scale_range, scale_range])
    fontsize = plot_parameters.get('FontSize', 15)

    # Sort events for plotting
    sorted_events = sorted([-int(re.findall(r'\d+', ev)[0]) for ev in top_events if re.search(r'\d+', ev)])
    sorted_events = [f"Event {abs(ev)}" for ev in sorted_events]
    sorted_events.append('Other Events')
    sorted_events.append('Pruned Events')

    # Define the Altair plot
    c = alt.Chart().encode(
        y=alt.Y('Event:O', sort=sorted_events, axis=alt.Axis(domain=False, labelFontSize=fontsize, title=None)),
    )

    a = c.mark_rect().encode(
        x=alt.X('Output Point Multiplied:O', axis=alt.Axis(titleFontSize=fontsize, title='Output Points')),
        color=alt.Color('rounded', title=None,
                        legend=alt.Legend(gradientLength=height, gradientThickness=10, orient='right', labelFontSize=fontsize),
                        scale=alt.Scale(domain=axis_lims, range=c_range))
    )

    b = c.mark_text(align='center', baseline='middle', dy=0, fontSize=fontsize,
                    color='#798184').encode(
        x=alt.X('Output Point Multiplied:O'),
        text='rounded_str',
    )

    cell_plot = alt.layer(a, b, data=expanded_data).properties(
        width=math.ceil(0.8 * width),
        height=height
    )

    return cell_plot

