
# 
import scipy.stats as stats
import os 
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from statsmodels.stats.multitest import multipletests
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import numpy as np
import plotly.graph_objects as go

current_dir = os.getcwd()

LIWC_list = [
        "LIWC_Negate", "LIWC_Swear", "LIWC_Social", "LIWC_Family",
        "LIWC_Friends", "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo",
        "LIWC_Anx", "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech", "LIWC_Sexual",
        "LIWC_Relativ", "LIWC_Achiev", "LIWC_Leisure", "LIWC_Money",
        "LIWC_Relig", "LIWC_Death"
    ]


def calculate_liwc_cluster_association(df,cluster_column='cluster_source', alpha=0.05):
    """
    Calculate association between clusters and LIWC categories using Chi-square tests.
    
    Parameters:
    - df: DataFrame containing cluster assignments and LIWC features
    - liwc_columns: List of LIWC column names
    - cluster_column: Column name for cluster assignments
    - alpha: Significance level for multiple testing correction
    """
    
    # Create binarized LIWC features (presence/absence or above/below median)
    liwc_binary = {}
    for liwc_col in LIWC_list:
        if liwc_col in df.columns:
            # Convert to binary: 1 if above median, 0 otherwise
            median_val = df[liwc_col].median()
            liwc_binary[liwc_col] = (df[liwc_col] > median_val).astype(int)
    
    # Get unique clusters
    clusters = sorted(df[cluster_column].unique())
    
    # Initialize results matrices
    p_values = np.ones((len(clusters), len(LIWC_list)))
    chi2_stats = np.zeros((len(clusters), len(LIWC_list)))
    effect_sizes = np.zeros((len(clusters), len(LIWC_list)))
    
    # Perform chi-square tests for each cluster-LIWC pair
    for i, cluster in enumerate(clusters):
        cluster_mask = (df[cluster_column] == cluster)
        
        for j, liwc_col in enumerate(LIWC_list):
            if liwc_col in df.columns:
                # Create contingency table
                contingency = pd.crosstab(
                    cluster_mask, 
                    liwc_binary[liwc_col]
                )
                
                # Ensure table is 2x2
                if contingency.shape == (2, 2):
                    try:
                        chi2, p, dof, expected = chi2_contingency(contingency)
                        p_values[i, j] = p
                        chi2_stats[i, j] = chi2
                        
                        # Calculate Cramér's V as effect size
                        n = contingency.sum().sum()
                        effect_sizes[i, j] = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
                    except:
                        p_values[i, j] = 1.0
                        effect_sizes[i, j] = 0.0
    
    # Apply multiple testing correction
    p_flat = p_values.flatten()
    reject_flat, p_corrected_flat, _, _ = multipletests(p_flat, alpha=alpha, method='fdr_bh')
    p_values_corrected = p_corrected_flat.reshape(p_values.shape)
    reject = reject_flat.reshape(p_values.shape)

        
    return {
        'p_values': p_values,
        'p_values_corrected': p_values_corrected,
        'chi2_stats': chi2_stats,
        'effect_sizes': effect_sizes,
        'reject_null': reject,
        'clusters': clusters,
        'liwc_columns': LIWC_list
    }

def create_combined_interactive_heatmap(results, 
                                        title="Association between cluster and LIWC"):

    effect = results['effect_sizes']
    p_corr = results['p_values_corrected']
    chi2_stats = results['chi2_stats']
    
    clusters = [f"Cluster {c}" for c in results['clusters']]
    liwc_columns = results['liwc_columns']

    # Significance annotation (stars)
    stars = np.where(
        p_corr < 0.001, "***",
        np.where(p_corr < 0.01, "**",
        np.where(p_corr < 0.05, "*", ""))
    )

    # Build hover text
    hover_text = []
    for i in range(effect.shape[0]):
        row = []
        for j in range(effect.shape[1]):
            star = stars[i, j]
            p_value = p_corr[i, j]
            effect_size = effect[i, j]
            chi2 = chi2_stats[i, j]

            row.append(
                f"<b>{clusters[i]} — {liwc_columns[j]}</b><br>"
                f"Corrected p-value: {p_value:.4e}<br>"
                f"Effect size (Cramér's V): {effect_size:.3f}<br>"
                f"Chi²: {chi2:.2f}<br>"
                f"Significance: {star if star!='' else 'ns'}"
            )
        hover_text.append(row)

    # Main heatmap (effect size)
    fig = go.Figure(data=go.Heatmap(
        z=effect,
        x=liwc_columns,
        y=clusters,
        colorscale="Viridis",
        text=stars,           # stars displayed on heatmap
        texttemplate="%{text}",
        hoverinfo="text",
        textfont={"color": "white", "size": 12},
        customdata=hover_text,
        hovertemplate="%{customdata}<extra></extra>",
        colorbar=dict(title="Cramér’s V")
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis=dict(title="LIWC Categories", tickangle=45),
        yaxis=dict(title="Clusters"),
        width=1200,
        height=700,
        plot_bgcolor="white"
    )

        # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Save HTML file
    file_path = os.path.join(current_dir, 'images', 'cross_cluster_conflict.html')
    fig.write_html(
        file_path)

    return fig

def print_significativity(results):
    significant_associations = np.sum(results['reject_null'])

    # Show most significant associations
    if significant_associations > 0:
        print(f" MOST SIGNIFICANT ASSOCIATIONS:")
        flat_indices = np.argsort(results['p_values_corrected'].flatten())
        for idx in flat_indices[:10]:  # Top 5
            i, j = np.unravel_index(idx, results['p_values_corrected'].shape)
            if results['p_values_corrected'][i, j] < 0.05:
                cluster = results['clusters'][i]
                liwc = results['liwc_columns'][j]
                p_val = results['p_values_corrected'][i, j]
                effect = results['effect_sizes'][i, j]
                print(f"  Cluster {cluster} - {liwc}: p={p_val:.6f}, effect={effect:.3f}")

def compute_cluster_conflict_matrix(df, cluster_col_source="cluster_source", cluster_col_target="cluster_target"):
    """
    Computes a symmetric matrix of negative interaction intensity
    between clusters.
    """

    # Filter only negative interactions
    df_neg = df[df["LINK_SENTIMENT"] == -1]

    clusters = sorted(df[cluster_col_source].dropna().unique())
    n = len(clusters)

    # create mapping cluster -> index
    idx = {c: i for i, c in enumerate(clusters)}

    # Empty square matrix
    M = np.zeros((n, n), dtype=int)

    # Fill matrix
    for _, row in df_neg.iterrows():
        s = row[cluster_col_source]
        t = row[cluster_col_target]
        if pd.notna(s) and pd.notna(t):
            M[idx[s], idx[t]] += 1
    
    # Make symmetric version
    M_sym = M + M.T  

    return clusters, M, M_sym