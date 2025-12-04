import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np


def visualize_liwc_recovery(df_timeseries, interesting_features):
    """
    LIWC feature recovery analysis
    
    Parameters
    ----------
    df_timeseries : DataFrame
        Output from delta_LIWC_timeseries function with multiple time segments
    interesting_liwc_features : list, optional
        List of specific LIWC features to focus on. If None, shows all.
    """
    
    delta_cols = [col for col in df_timeseries.columns if col.startswith('delta_LIWC_')]
    
    if interesting_features:
        # Filter to only include specified interesting features
        delta_cols = [col for col in delta_cols if any(feature in col for feature in interesting_features)]
    
    # Calculate recovery metrics for each feature
    recovery_data = []
    
    for feature in delta_cols:
        feature_name = feature.replace('delta_LIWC_', '')
        
        # Calculate recovery times
        recovery_times = []
        never_recovered = 0
        
        # Group by each conflict (target + t0)
        for (target, t0), group in df_timeseries.groupby(['target', 't0']):
            group = group.sort_values('window_start_h')
            
            # Find the first window where delta <= 0 (returned to baseline)
            recovered_rows = group[group[feature] <= 0]
            
            if not recovered_rows.empty:
                # Use the end time of the first recovery window
                recovery_time = recovered_rows.iloc[0]['window_end_h']
                recovery_times.append(recovery_time)
            else:
                # Feature never recovered within observation window
                never_recovered += 1
                recovery_times.append(48)  # Use max window as "never recovered"
        
        # Convert to Series for calculations
        recovery_series = pd.Series(recovery_times)
        
        # Calculate overall statistics across all time segments
        mean_delta = df_timeseries[feature].mean()
        median_delta = df_timeseries[feature].median()
        std_delta = df_timeseries[feature].std()
        
        # Calculate recovery metrics
        total_cases = len(recovery_times)
        recovered_within_48h = total_cases - never_recovered
        recovery_rate = recovered_within_48h / total_cases if total_cases > 0 else 0
        
        # Average recovery time (only for those that actually recovered)
        actual_recovery_times = [t for t in recovery_times if t < 48]
        avg_recovery_time = np.mean(actual_recovery_times) if actual_recovery_times else 48
        
        # Median recovery time
        median_recovery_time = np.median(actual_recovery_times) if actual_recovery_times else 48
        
        recovery_data.append({
            'feature': feature_name,
            'mean_delta': mean_delta,
            'median_delta': median_delta,
            'std_delta': std_delta,
            'recovery_rate': recovery_rate,
            'avg_recovery_time_hours': avg_recovery_time,
            'median_recovery_time_hours': median_recovery_time,
            'never_recovered_count': never_recovered,
            'total_cases': total_cases,
            'direction': 'increase' if mean_delta > 0 else 'decrease',
            'effect_size': abs(mean_delta) / std_delta if std_delta > 0 else 0,
            'recovery_times': recovery_times  # Store for plotting
        })
    
    recovery_df = pd.DataFrame(recovery_data)
    
    # PLOT 1: Delta LIWC
    fig1 = go.Figure()
    
    colors = ['red' if x > 0 else 'blue' for x in recovery_df['mean_delta']]
    fig1.add_trace(
        go.Bar(x=recovery_df['feature'], y=recovery_df['mean_delta'],
               name='Mean Change', marker_color=colors,
               error_y=dict(type='data', array=recovery_df['std_delta'], visible=True))
    )
    
    fig1.update_layout(
        height=500,
        showlegend=False,
        title_text="LIWC Feature Changes After Conflict",
        template="plotly_white",
        xaxis_tickangle=45
    )
    
    fig1.update_yaxes(title_text="Mean Δ LIWC")
    
    fig1.show()
    
    # Save the first plot as HTML
    fig1.write_html(
        'images/liwc_delta_analysis.html',
        include_plotlyjs=True,
        config={
            'responsive': True,
            'displayModeBar': True
        },
        auto_open=False
    )

    # PLOT 2: Recovery Rate
    fig2 = go.Figure()
    
    fig2.add_trace(
        go.Bar(x=recovery_df['feature'], y=recovery_df['recovery_rate'],
               name='Recovery Rate', marker_color='green')
    )
    
    fig2.update_layout(
        height=500,
        showlegend=False,
        title_text="LIWC Feature Recovery Rate",
        template="plotly_white",
        xaxis_tickangle=45
    )
    
    fig2.update_yaxes(title_text="Recovery Rate", range=[0, 1])
    
    fig2.show()
    
    # Save the second plot as HTML
    fig2.write_html(
        'images/liwc_recovery_rate.html',
        include_plotlyjs=True,
        config={
            'responsive': True,
            'displayModeBar': True
        },
        auto_open=False
    )

    # Top 5 features with largest increases
    top_increases = recovery_df.nlargest(5, 'mean_delta')
    print("\nTOP 5 FEATURES WITH LARGEST INCREASES:")
    for _, row in top_increases.iterrows():
        print(f"  • {row['feature']}: +{row['mean_delta']:.4f} (Recovery: {row['avg_recovery_time_hours']:.1f}h)")
    
    # Top 5 features with largest decreases  
    top_decreases = recovery_df.nsmallest(5, 'mean_delta')
    print("\n TOP 5 FEATURES WITH LARGEST DECREASES:")
    for _, row in top_decreases.iterrows():
        print(f"  • {row['feature']}: {row['mean_delta']:.4f} (Recovery: {row['avg_recovery_time_hours']:.1f}h)")
    
    # Fastest recovering features
    fastest_recovery = recovery_df.nsmallest(5, 'avg_recovery_time_hours')
    print("\n FASTEST RECOVERING FEATURES:")
    for _, row in fastest_recovery.iterrows():
        print(f"  • {row['feature']}: {row['avg_recovery_time_hours']:.1f}h to recover")
    
    # Features that never recover
    never_recover = recovery_df.nlargest(5, 'never_recovered_count')
    print("\n FEATURES THAT OFTEN DON'T RECOVER:")
    for _, row in never_recover.iterrows():
        if row['never_recovered_count'] > 0:
            print(f"  • {row['feature']}: {row['never_recovered_count']}/{row['total_cases']} never recovered")
    
    return recovery_df
