
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd

def plot_recovery_histogram(recovery_df, liwc_features=None):
    """
    Histogram with statistics and better styling
    """
    
    if liwc_features is None:
        liwc_features = recovery_df['feature'].tolist()
    
    # Calculate statistics for each feature
    stats_data = []
    for feature in liwc_features:
        feature_row = recovery_df[recovery_df['feature'] == feature]
        if len(feature_row) > 0:
            recovery_times = feature_row['recovery_times'].iloc[0]
            stats = {
                'feature': feature,
                'recovery_times': recovery_times,
                'never_recovered': (np.array(recovery_times) == 48).sum(),
                'recovered_within_24h': (np.array(recovery_times) <= 24).sum(),
                'total_cases': len(recovery_times),
                'avg_recovery_time': np.mean([t for t in recovery_times if t < 48]) if any(t < 48 for t in recovery_times) else 48
            }
            stats['never_recovered_pct'] = stats['never_recovered'] / stats['total_cases']
            stats['recovered_24h_pct'] = stats['recovered_within_24h'] / stats['total_cases']
            stats_data.append(stats)
    
    stats_df = pd.DataFrame(stats_data)
    
    # Create figure
    fig = go.Figure()
    
    # Default feature
    default_feature = liwc_features[0]
    default_stats = stats_df[stats_df['feature'] == default_feature].iloc[0]
    
    fig.add_trace(go.Histogram(
        x=default_stats['recovery_times'],
        nbinsx=24,
        name=default_feature,
        opacity=0.7,
        marker_color='lightblue',
        hovertemplate='<b>Recovery Time: %{x}h</b><br>Number of Conflicts: %{y}<extra></extra>'
    ))
    
    # Create dropdown with statistics
    dropdown_buttons = []
    for _, stats in stats_df.iterrows():
        label = f"{stats['feature']} ({stats['recovered_24h_pct']:.0%} in 24h)"
        
        dropdown_buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"x": [stats['recovery_times']]},
                    {
                        "title": f"Recovery Time - {stats['feature']}",
                        "annotations": [
                            dict(
                                x=0.5, y=1.08, xref="paper", yref="paper",
                                text=f"Avg: {stats['avg_recovery_time']:.1f}h | Never recovered: {stats['never_recovered']}/{stats['total_cases']} ({stats['never_recovered_pct']:.1%})",
                                showarrow=False,
                                font=dict(size=12, color="red"),
                                align="center"
                            )
                        ]
                    }
                ]
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"Recovery Time Distribution - {default_feature}",
        xaxis_title="Hours to Recovery (delta ≤ 0)",
        yaxis_title="Number of Conflicts",
        xaxis=dict(
            range=[0, 48],
            dtick=6,
            tickvals=list(range(0, 49, 6)),
            ticktext=[f"{h}h" for h in range(0, 49, 6)]
        ),
        template="plotly_white",
        height=550,
        annotations=[
            dict(
                x=0.5, y=1.08, xref="paper", yref="paper",
                text=f"Avg: {default_stats['avg_recovery_time']:.1f}h | Never recovered: {default_stats['never_recovered']}/{default_stats['total_cases']} ({default_stats['never_recovered_pct']:.1%})",
                showarrow=False,
                font=dict(size=12, color="red"),
                align="center"
            )
        ],
        updatemenus=[dict(
            buttons=dropdown_buttons,
            direction="down",
            showactive=True,
            x=0.85,
            xanchor="left",
            y=1.0,
            yanchor="top",
            bgcolor="lightgray"
        )]
    )
    
    # Add reference lines
    fig.add_vline(x=24, line_dash="dash", line_color="orange", 
                  annotation_text="24h", annotation_position="top right")
    fig.add_vline(x=48, line_dash="dash", line_color="red", 
                  annotation_text="48h (max)", annotation_position="top right")
    
    fig.show()
    
    # Save the histogram plot as HTML
    fig.write_html(
        'images/recovery_histogram.html',
        include_plotlyjs=True,
        config={
            'responsive': True,
            'displayModeBar': True
        },
        auto_open=False
    )

    return stats_df
