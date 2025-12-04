from scipy.stats import pearsonr
import warnings
import plotly.express as px
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

def analyze_attacker_impact_timeseries(df_timeseries, properties_name):
    """
    Analyze attacker impact across all time segments combined
    """
    impact_data = []
    
    for _, row in df_timeseries.iterrows():
        record = {
            'source': row['source'],
            'target': row['target'],
            't0': row['t0'],
            'window_start_h': row['window_start_h'],
            'window_end_h': row['window_end_h'],
            'segment_label': row['segment_label']
        }
        
        # Attacker features (incoming properties)
        for prop in properties_name:
            record[f'attacker_{prop}'] = row[f'incoming_LIWC_{prop}']
        
        # Target delta features (changes in target behavior)
        for prop in properties_name:
            record[f'target_delta_{prop}'] = row[f'delta_LIWC_{prop}']
        
        impact_data.append(record)
    
    return pd.DataFrame(impact_data)

def compute_correlations_overall(impact_df, properties_name):
    """
    Compute overall correlations between attacker features and target changes
    """
    results = []

    for attacker_feat in properties_name:
        attacker_col = f'attacker_{attacker_feat}'
        
        for target_feat in properties_name:
            target_col = f'target_delta_{target_feat}'
                
            clean_data = impact_df[[attacker_col, target_col]].dropna()
            
            if len(clean_data) > 10:  
                try:
                    corr, p_value = pearsonr(clean_data[attacker_col], clean_data[target_col])
                    
                    results.append({
                        'attacker': attacker_feat,
                        'target': target_feat,
                        'correlation': corr,
                        'p_value': p_value,
                        'n_samples': len(clean_data),
                        'abs_corr': abs(corr)
                    })
                except:
                    continue
    
    corr_df = pd.DataFrame(results)
    if len(corr_df) > 0:
        corr_df = corr_df.sort_values('abs_corr', ascending=False)
    
    return corr_df

def visualize_results(corr_df):
    """
    Visualize overall correlation results 
    """
    
    top_corr = corr_df
    
    try:
        heatmap_data = top_corr.pivot(
            index='attacker', 
            columns='target', 
            values='correlation'
        )
        
        fig1 = px.imshow(
            heatmap_data,
            title='<b>Top Overall Correlations: Attacker Features → Target Changes</b>',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        fig1.show()
        fig1.write_html(
        'images/attack_LIWC_correlation.html',
        include_plotlyjs=True,
        config={
            'responsive': True,
            'displayModeBar': True
        },
        auto_open=False
    )
    except Exception as e:
        print(f"Could not create heatmap: {e}")
    
    # Attacker power ranking
    attacker_power = corr_df.groupby('attacker')['abs_corr'].mean().sort_values(ascending=False)
    
    if len(attacker_power) > 0:
        fig2 = px.bar(
            x=attacker_power.head(15).values,
            y=attacker_power.head(15).index,
            orientation='h',
            title='<b>Most Influential Attacker Features </b>',
            labels={'x': 'Average Impact (Absolute Correlation)', 'y': 'Attacker Feature'}
        )
        fig2.show()
        fig2.write_html(
        'images/influential-attack.html',
        include_plotlyjs=True,
        config={
            'responsive': True,
            'displayModeBar': True
        },
        auto_open=False
    )


    return attacker_power

def attacker_impact_analysis(df_timeseries, properties_name):
    """
    Complete attacker impact analysis for all timestamps combined
    """

    # Analyze impact data
    impact_df = analyze_attacker_impact_timeseries(df_timeseries, properties_name)

    # Compute overall correlations
    corr_df = compute_correlations_overall(impact_df, properties_name)
    
    if len(corr_df) > 0:

        # Visualize results
        attacker_power = visualize_results(corr_df)
        
        # Most powerful attacker feature
        if not corr_df.empty:
            best_attacker = corr_df.groupby('attacker')['abs_corr'].mean().idxmax()
            best_corr_value = corr_df.groupby('attacker')['abs_corr'].mean().max()
            print(f"\n Most influential attacker feature: '{best_attacker}' (avg |r| = {best_corr_value:.3f})")
            
            # Show top 5 correlations
            print(f"\n Top 5 attacker→target correlations:")
            for i in range(min(5, len(corr_df))):
                row = corr_df.iloc[i]
                print(f"   {i+1}. {row['attacker']} → {row['target']}: r = {row['correlation']:.3f} (p = {row['p_value']:.4f})")
        else:
            best_attacker = None
            print(" No influential features found")
        
    else:
        print("No significant correlations found")
        attacker_power = None
        best_attacker = None
    
    return {
        'impact_data': impact_df,
        'correlations': corr_df,
        'attacker_power': attacker_power,
        'best_attacker': best_attacker
    }
