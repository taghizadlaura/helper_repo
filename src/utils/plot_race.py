import plotly.express as px
import pandas as pd

def create_cluster_race_plot(
        df,
        timestamp_col='TIMESTAMP',
        group_col='cluster_source',
        value_col=None,
        period='M',
        output_html='images/race_plot.html'
    ):
    """
    Create an animated bar race plot showing cluster metrics over time.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least a timestamp column and a cluster column.
    timestamp_col : str, default 'TIMESTAMP'
        Name of the timestamp column.
    group_col : str, default 'cluster_source'
        Column representing the cluster/group identifier.
    value_col : str or None, default None
        Column to aggregate. 
        If None → counts rows (message count).
    period : str, default 'M'
        Pandas period frequency, e.g.:
        'M' monthly, 'D' daily, 'W' weekly, 'Q' quarterly.
    output_html : str or None, default 'images/race_plot.html'
        Path to save HTML output. If None, file is not saved.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        The animated Plotly bar race figure.
    """

    # --- Validate columns ---
    for col in [timestamp_col, group_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    # --- Prepare data ---
    df_copy = df.copy()

    # Ensure timestamp format
    if not pd.api.types.is_datetime64_any_dtype(df_copy[timestamp_col]):
        df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])

    # Create period label
    df_copy['period'] = df_copy[timestamp_col].dt.to_period(period).astype(str)

    # Aggregate values
    if value_col is None:
        # Count occurrences (default behaviour)
        agg_df = (
            df_copy.groupby(['period', group_col])
            .size()
            .reset_index(name='metric_value')
        )
        metric_label = "Message Count"
    else:
        if value_col not in df_copy.columns:
            raise ValueError(f"value_col '{value_col}' not in DataFrame")

        agg_df = (
            df_copy.groupby(['period', group_col])[value_col]
            .sum()
            .reset_index(name='metric_value')
        )
        metric_label = value_col

    # Sort for better animation transitions
    agg_df = agg_df.sort_values(['period', 'metric_value'], ascending=[True, False])

    # --- Create animated chart ---
    fig = px.bar(
        agg_df,
        x='metric_value',
        y=group_col,
        animation_frame='period',
        orientation='h',
        color=group_col,
        title=f"{metric_label} Evolution Over Time",
        labels={
            'metric_value': metric_label,
            group_col: "Cluster",
            'period': "Time Period"
        },
        height=600
    )

    # Layout improvements
    fig.update_layout(
        xaxis_title=metric_label,
        yaxis_title="Cluster",
        showlegend=False,
        font=dict(size=12)
    )

    # Animation speed (if exists)
    if fig.layout.updatemenus:
        btn = fig.layout.updatemenus[0].buttons[0]
        btn.args[1]["frame"]["duration"] = 1000     # 1 second per frame
        btn.args[1]["transition"]["duration"] = 400

    # Display figure
    fig.show()

    # Save HTML output
    if output_html:
        fig.write_html(
            output_html,
            include_plotlyjs=True,
            config={'responsive': True, 'displayModeBar': True},
            auto_open=False
        )

    return fig
