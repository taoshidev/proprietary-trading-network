import marimo

__generated_with = "0.8.14"
app = marimo.App(width="full")


@app.cell
def __():
    from matplotlib import pyplot as plt
    import argparse
    import pandas as pd
    from pandas import json_normalize

    # Set the display options for Pandas
    pd.set_option('display.max_rows', 1000)  # Set to a high number as needed
    pd.set_option('display.max_columns', 1000)  # Set to a high number as needed
    import math
    return argparse, json_normalize, math, pd, plt


@app.cell
def __():
    import marimo as mo
    import altair as alt
    return alt, mo


@app.cell
def __():
    from IPython.display import display, HTML

    # Inject JavaScript to handle the click event
    display(HTML('''
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const charts = document.querySelectorAll('a[href]');
        charts.forEach(chart => {
            chart.setAttribute('target', '_blank');
        });
    });
    </script>
    '''))
    return HTML, display


@app.cell
def __():
    moo = 'thing'
    return moo,


@app.cell
def __(alt):
    # Define a base theme with slightly smaller fonts
    font_size = 13
    medium_font_theme = {
        'config': {
            'title': {'fontSize': font_size + 4},
            'axis': {
                'labelFontSize': font_size,
                'titleFontSize': font_size + 2,
            },
            'legend': {
                'labelFontSize': font_size,
                'titleFontSize': font_size + 2,
            },
            'tooltip': {
                'fontSize': font_size
            }
        }
    }

    # Apply the custom theme
    alt.themes.register('medium_font', lambda: medium_font_theme)
    alt.themes.enable('medium_font')
    return font_size, medium_font_theme


@app.cell
def __():
    a = 109
    return a,


@app.cell
def __():
    import sys
    import os
    return os, sys


@app.cell
def __():
    import time_util
    return time_util,


@app.cell
def __(time_util):
    time_util
    return


@app.cell
def __():
    from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
    subtensor_weight_setter = SubtensorWeightSetter(None, None, None)
    return SubtensorWeightSetter, subtensor_weight_setter


@app.cell
def __():
    from runnable.generate_request_minerstatistics import generate_miner_statistics_data
    return generate_miner_statistics_data,


@app.cell
def __():
    time_now = 1723175571636 # TimeUtil.now_in_millis()
    return time_now,


@app.cell
def __(generate_miner_statistics_data, time_now):
    miner_statistics = generate_miner_statistics_data(time_now=time_now, checkpoints=False)
    return miner_statistics,


@app.cell
def __(miner_statistics):
    miner_statistics_data = miner_statistics['data']
    del miner_statistics['data']
    return miner_statistics_data,


@app.cell
def __(miner_statistics_data, subtensor_weight_setter):
    miner_statistics_hotkeys = [ x['hotkey'] for x in miner_statistics_data ]
    filtered_ledger = subtensor_weight_setter.filtered_ledger(hotkeys=miner_statistics_hotkeys)
    filtered_positions = subtensor_weight_setter.filtered_positions(hotkeys=miner_statistics_hotkeys)
    return filtered_ledger, filtered_positions, miner_statistics_hotkeys


@app.cell
def __(json_normalize, miner_statistics_data):
    miner_statistics_dataframe = json_normalize(miner_statistics_data, sep='_')
    return miner_statistics_dataframe,


@app.cell
def __():
    return


@app.cell
def __(miner_statistics_dataframe):
    miner_results = miner_statistics_dataframe
    # Add a column for the URL
    # Add a column for the URL and reset index
    miner_results = miner_results.reset_index()
    miner_results['url'] = 'https://dashboard.taoshi.io/miner/' + miner_results['index'].astype(str)

    # Update index to only show last 20 characters for tooltip display
    miner_results['short_index'] = miner_results['index'].astype(str).str[-10:]
    return miner_results,


@app.cell
def __(miner_results):
    miner_results
    return


@app.cell
def __(alt, miner_results, mo):
    # Scatter Plot of Original Scores with uniform color scheme for ranking
    scatter_plot_original_scores = alt.Chart(miner_results).mark_point(size=100, filled=True, opacity=0.75).encode(
        x='scores_risk_adjusted_return_value',
        y='scores_short_risk_adjusted_return_value',
        color=alt.Color('weight_value:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=[
            alt.Tooltip('short_index:O', title='Index'),
            alt.Tooltip('scores_risk_adjusted_return_value', title='Long Score'),
            alt.Tooltip('scores_short_risk_adjusted_return_value', title='Long Unpenalized Score'),
            alt.Tooltip('weight_value', title='Final Rank'),
        ],
        href='url:N'
    ).properties(
        title='Scatter Plot of Original Scores',
        height=500  # Increased height for more vertical space
    ).transform_calculate(
        url='datum.url + "?target=_blank"'
    ).encode(
        x='scores_risk_adjusted_return_value',
        y='scores_short_risk_adjusted_return_value',
        color='weight_value:Q'
    )

    chart = mo.ui.altair_chart(scatter_plot_original_scores)
    return chart, scatter_plot_original_scores


@app.cell
def __(chart):
    chart
    return


@app.cell
def __(alt, miner_results, mo):
    # Scatter Plot of Long Returns vs Omega, colored by Weight
    scatter_plot_long_omega = alt.Chart(miner_results).mark_point(size=100).encode(
        x='scores_risk_adjusted_return_value',
        y='scores_omega_value',
        color=alt.Color('scores_risk_adjusted_return_value:Q', scale=alt.Scale(scheme='plasma')),
        tooltip=[
            alt.Tooltip('short_index:O', title='Index'),
            alt.Tooltip('scores_risk_adjusted_return_value', title='Long Score'),
            alt.Tooltip('scores_omega_value', title='Omega'),
            alt.Tooltip('weight_rank:Q', title='Weight'),
        ],
        href='url:N'
    ).properties(
        title='Scatter Plot of Long Returns vs Omega',
        height=400  # Adjusted height
    )

    mo.ui.altair_chart(scatter_plot_long_omega)
    return scatter_plot_long_omega,


@app.cell
def __(alt, miner_results, mo):
    # Scatter Plot of Long Returns vs Sharpe, colored by Weight
    scatter_plot_long_sharpe = alt.Chart(miner_results).mark_point(size=100).encode(
        x='scores_risk_adjusted_return_value',
        y='scores_sharpe_value',
        color=alt.Color('weight_rank:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=[
            alt.Tooltip('short_index:O', title='Index'),
            alt.Tooltip('scores_risk_adjusted_return_value', title='Long Score'),
            alt.Tooltip('scores_sharpe_value', title='Sharpe'),
            alt.Tooltip('weight_rank:Q', title='Weight'),
        ],
        href='url:N'
    ).properties(
        title='Scatter Plot of Long Returns vs Sharpe',
        width=600,
        height=400  # Adjusted height
    )

    mo.ui.altair_chart(scatter_plot_long_sharpe)
    return scatter_plot_long_sharpe,


@app.cell
def __(miner_results):
    filtered_miner_results = miner_results[miner_results['penalties_total'] > 0]
    return filtered_miner_results,


@app.cell
def __(alt, filtered_miner_results, mo):
    # Heatmap of Penalties with index on hover
    penalties_df_melted = filtered_miner_results[[
        'short_index', 
        'penalties_time_consistency', 
        'penalties_returns_ratio', 
        'penalties_drawdown_threshold', 
        'penalties_drawdown', 
        'penalties_daily', 
        'penalties_biweekly', 
        'penalties_total'
    ]].melt(id_vars=['short_index'], var_name='Penalty Type', value_name='Value')

    penalties_df_melted['url'] = 'https://dashboard.taoshi.io/miner/' + filtered_miner_results['index'].astype(str)

    heatmap_penalties = alt.Chart(penalties_df_melted).mark_rect().encode(
        x='Penalty Type:O',
        y='short_index:O',
        color=alt.Color('Value:Q', scale=alt.Scale(scheme='inferno')),
        tooltip=[
            alt.Tooltip('short_index:O', title='Index'),
            alt.Tooltip('Penalty Type', title='Penalty Type'),
            alt.Tooltip('Value:Q', title='Value'),
            alt.Tooltip('url:N', title='Link')
        ],
        href='url:N'
    ).properties(
        title='Heatmap of Penalties'
    ).transform_calculate(
        url='datum.url + "?target=_blank"'
    )

    mo.ui.altair_chart(heatmap_penalties)
    return heatmap_penalties, penalties_df_melted


@app.cell
def __():
    # # Box Plot of Penalized Risk-Adjusted Return Value across Penalized Rank
    # box_plot_scores_ranks = alt.Chart(miner_results).mark_boxplot().encode(
    #     x='penalized_scores_risk_adjusted_return_rank:O',
    #     y='penalized_scores_risk_adjusted_return_value:Q',
    #     color=alt.Color('penalized_scores_risk_adjusted_return_rank:Q', scale=alt.Scale(scheme='viridis')),
    #     tooltip=['penalized_scores_risk_adjusted_return_rank', 'penalized_scores_risk_adjusted_return_value']
    # ).properties(
    #     title='Box Plot of Penalized Risk-Adjusted Return Value across Penalized Rank'
    # )

    # box_chart = mo.ui.altair_chart(box_plot_scores_ranks)
    # box_chart
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
