import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_cluster_race_plot(df_title, timestamp_col='TIMESTAMP'):
    """
    Create an interactive bar race plot showing cluster message volume through time
    
    Parameters:
    df_title: DataFrame containing the data
    timestamp_col: Name of the timestamp column
    """
    
    # Prepare data: count messages per cluster per time period
    df_time_series = df_title.groupby([timestamp_col, 'cluster_source']).size().reset_index(name='message_count')
    
    # Convert timestamp to proper datetime if needed and extract month/year
    if not pd.api.types.is_datetime64_any_dtype(df_time_series[timestamp_col]):
        df_time_series[timestamp_col] = pd.to_datetime(df_time_series[timestamp_col])
    
    # Create a period column (month-year)
    df_time_series['period'] = df_time_series[timestamp_col].dt.to_period('M').astype(str)
    
    # Aggregate by period and cluster
    df_period = df_time_series.groupby(['period', 'cluster_source'])['message_count'].sum().reset_index()
    
    # Sort by period and message count for better visualization
    df_period = df_period.sort_values(['period', 'message_count'], ascending=[True, False])
    
    # Create the animated bar chart
    fig = px.bar(
        df_period,
        x='message_count',
        y='cluster_source',
        animation_frame='period',
        orientation='h',
        color='cluster_source',
        title='Cluster Message Volume Evolution Over Time',
        labels={
            'message_count': 'Number of Messages',
            'cluster_source': 'Cluster ID',
            'period': 'Time Period'
        },
        height=600
    )
    
    # Update layout for better visualization
    fig.update_layout(
        xaxis_title="Number of Messages",
        yaxis_title="Cluster ID",
        showlegend=False,
        font=dict(size=12)
    )
    
    # Adjust animation settings
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500
    
    fig.show()

    fig.write_html(
    'images/race_plot.html',
            include_plotlyjs=True,
            config={
                'responsive': True,
                'displayModeBar': True
            },
            auto_open=False)
