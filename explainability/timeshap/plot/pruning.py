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

import numpy as np
import pandas as pd
import altair as alt
import copy


def plot_temp_coalition_pruning(df: pd.DataFrame,
                                pruned_idx: int,
                                plot_limit: int = 50,
                                solve_negatives: bool = True,
                                ):
    """Plots the coalition explainer process

    Parameters
    ----------
    df: pd.DataFrame
        Pruning algorithm output

    pruned_idx: int
        Index the explainer takes place

    plot_limit: int
        Window of events to show the explainer to
        Default is 50

    solve_negatives: bool
        Whether to remove negative importances of the background instances
    """

    def solve_negatives_method(df):
        negative_values = copy.deepcopy(df[df['Shapley Value'] < 0])
        for idx, row in negative_values.iterrows():
            corresponding_row = df[np.logical_and(df['t (event index)'] == row['t (event index)'], ~(df['Coalition'] == row['Coalition']))]
            df.at[corresponding_row.index[0], 'Shapley Value'] = corresponding_row['Shapley Value'].values[0] + row['Shapley Value']
            df.at[idx, 'Shapley Value'] = 0
        return df

    df = df[df['t (event index)'] >= -plot_limit]
    if solve_negatives:
        df = solve_negatives_method(df)

    min_shapley_value = df['Shapley Value'].min()
    max_shapley_value = df['Shapley Value'].max()

    base = (alt.Chart(df).encode(
        x=alt.X("t (event index)", axis=alt.Axis(title='t (event index)', labelFontSize=15,
                              titleFontSize=15)),
        y=alt.Y("Shapley Value",
                axis=alt.Axis(titleFontSize=15, grid=True, labelFontSize=15,
                              titleX=-28),
                scale=alt.Scale(domain=[min_shapley_value, 1.2*max_shapley_value], )),
        color=alt.Color('Coalition', scale=alt.Scale(
            domain=['Sum of contribution of events \u2264 t'],
            range=["#618FE0"]), legend=alt.Legend(title=None, labelLimit=0,
                                                  fillColor="white",
                                                  labelFontSize=14,
                                                  symbolStrokeWidth=0,
                                                  symbolSize=50,
                                                  orient="top-left")),
    )
    )

    area_chart = base.mark_area(opacity=0.5)
    line_chart = base.mark_line()

    line = alt.Chart(pd.DataFrame({'x': -len(pruned_idx) + np.where(pruned_idx == 0)[0]})).mark_rule(
        color='#E17560').encode(x='x')

    pruning_graph = (line + area_chart + line_chart).properties(width=350,height=225)

    return pruning_graph
