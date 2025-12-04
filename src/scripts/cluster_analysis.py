import scipy.stats as stats
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from statsmodels.stats.multitest import multipletests

def calculate_liwc_cluster_association(df, liwc_columns, cluster_column='cluster_source', alpha=0.05):
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
    for liwc_col in liwc_columns:
        if liwc_col in df.columns:
            # Convert to binary: 1 if above median, 0 otherwise
            median_val = df[liwc_col].median()
            liwc_binary[liwc_col] = (df[liwc_col] > median_val).astype(int)
    
    # Get unique clusters
    clusters = sorted(df[cluster_column].unique())
    
    # Initialize results matrices
    p_values = np.ones((len(clusters), len(liwc_columns)))
    chi2_stats = np.zeros((len(clusters), len(liwc_columns)))
    effect_sizes = np.zeros((len(clusters), len(liwc_columns)))
    
    # Perform chi-square tests for each cluster-LIWC pair
    for i, cluster in enumerate(clusters):
        cluster_mask = (df[cluster_column] == cluster)
        
        for j, liwc_col in enumerate(liwc_columns):
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
        'liwc_columns': liwc_columns
    }

def create_pvalue_heatmap(results, title="Cluster-LIWC Association P-values"):
    """
    Create a heatmap of p-values between clusters and LIWC categories.
    """
    p_matrix = -np.log10(results['p_values_corrected'] + 1e-10)  # -log10(p) for better visualization
    clusters = results['clusters']
    liwc_columns = results['liwc_columns']
    
    # Create annotation matrix (show actual p-values)
    annotation_matrix = np.where(
        results['p_values_corrected'] < 0.001, '***',
        np.where(results['p_values_corrected'] < 0.01, '**',
        np.where(results['p_values_corrected'] < 0.05, '*', ''))
    )
    
    plt.figure(figsize=(max(12, len(liwc_columns)*0.8), max(8, len(clusters)*0.6)))
    
    # Create heatmap
    im = plt.imshow(p_matrix, cmap='YlOrRd', aspect='auto')
    
    # Add annotations
    for i in range(len(clusters)):
        for j in range(len(liwc_columns)):
            text = plt.text(j, i, annotation_matrix[i, j],
                           ha="center", va="center", color="black", fontweight='bold')
    
    # Customize plot
    plt.colorbar(im, label='-log10(p-value)')
    plt.xlabel('LIWC Categories', fontsize=12)
    plt.ylabel('Clusters', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Set ticks
    plt.xticks(range(len(liwc_columns)), liwc_columns, rotation=45, ha='right')
    plt.yticks(range(len(clusters)), [f'Cluster {c}' for c in clusters])
    
    plt.tight_layout()
    plt.show()
    
    return p_matrix

def create_interactive_pvalue_heatmap(results, title="Cluster-LIWC Association P-values"):
    """
    Create an interactive heatmap using Plotly.
    """
    p_matrix = -np.log10(results['p_values_corrected'] + 1e-10)
    clusters = [f'Cluster {c}' for c in results['clusters']]
    liwc_columns = results['liwc_columns']
    
    # Create hover text
    hover_text = []
    for i, cluster in enumerate(results['clusters']):
        row_text = []
        for j, liwc_col in enumerate(liwc_columns):
            p_val = results['p_values_corrected'][i, j]
            effect = results['effect_sizes'][i, j]
            chi2 = results['chi2_stats'][i, j]
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            
            text = (f"<b>Cluster {cluster} - {liwc_col}</b><br>"
                   f"p-value: {p_val:.6f}<br>"
                   f"-log10(p): {p_matrix[i, j]:.3f}<br>"
                   f"Effect size (Cramér's V): {effect:.3f}<br>"
                   f"Chi²: {chi2:.3f}<br>"
                   f"Significance: {significance}")
            row_text.append(text)
        hover_text.append(row_text)
    
    fig = go.Figure(data=go.Heatmap(
        z=p_matrix,
        x=liwc_columns,
        y=clusters,
        hoverinfo='text',
        text=hover_text,
        colorscale='YlOrRd',
        colorbar=dict(title='-log10(p-value)'),
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16, color='darkred')
        ),
        xaxis_title="LIWC Categories",
        yaxis_title="Clusters",
        width=1000,
        height=600,
        xaxis=dict(tickangle=45),
        plot_bgcolor='white'
    )
    
    return fig

def create_effect_size_heatmap(results, title="Cluster-LIWC Effect Sizes (Cramér's V)"):
    """
    Create a heatmap showing effect sizes.
    """
    effect_matrix = results['effect_sizes']
    clusters = results['clusters']
    liwc_columns = results['liwc_columns']
    
    # Create significance mask
    significance_mask = results['p_values_corrected'] < 0.05
    
    plt.figure(figsize=(max(12, len(liwc_columns)*0.8), max(8, len(clusters)*0.6)))
    
    # Create heatmap
    im = plt.imshow(effect_matrix, cmap='viridis', aspect='auto')
    
    # Add significance markers
    for i in range(len(clusters)):
        for j in range(len(liwc_columns)):
            if significance_mask[i, j]:
                plt.plot(j, i, 'w*', markersize=10)
    
    plt.colorbar(im, label="Cramér's V Effect Size")
    plt.xlabel('LIWC Categories', fontsize=12)
    plt.ylabel('Clusters', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.xticks(range(len(liwc_columns)), liwc_columns, rotation=45, ha='right')
    plt.yticks(range(len(clusters)), [f'Cluster {c}' for c in clusters])
    
    plt.tight_layout()
    plt.show()


LIWC_list = [
 "LIWC_Negate", "LIWC_Swear", "LIWC_Social", "LIWC_Family",
 "LIWC_Friends", "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo",
 "LIWC_Anx", "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech", "LIWC_Sexual",
 "LIWC_Relativ", "LIWC_Achiev", "LIWC_Leisure", "LIWC_Money",
 "LIWC_Relig", "LIWC_Death"
]


def cluster_analysis(df):
    results=calculate_liwc_cluster_association(df, LIWC_list, cluster_column='cluster_source', alpha=0.05)
    # Create visualizations
    p_matrix = create_pvalue_heatmap(results, "Statistical Significance: Cluster-LIWC Associations")
    interactive_fig = create_interactive_pvalue_heatmap(results)
    interactive_fig.show()
        
    create_effect_size_heatmap(results)

    significant_associations = np.sum(results['reject_null'])
    total_tests = results['reject_null'].size
    print(f"Total tests performed: {total_tests}")
    print(f"Significant associations (p < 0.05 after correction): {significant_associations}")
    print(f"Percentage significant: {significant_associations/total_tests*100:.1f}%")
        
    # Show most significant associations
    if significant_associations > 0:
        print(f"\ MOST SIGNIFICANT ASSOCIATIONS:")
        flat_indices = np.argsort(results['p_values_corrected'].flatten())
        for idx in flat_indices[:5]:  # Top 5
            i, j = np.unravel_index(idx, results['p_values_corrected'].shape)
            if results['p_values_corrected'][i, j] < 0.05:
                cluster = results['clusters'][i]
                liwc = results['liwc_columns'][j]
                p_val = results['p_values_corrected'][i, j]
                effect = results['effect_sizes'][i, j]
                print(f"  Cluster {cluster} - {liwc}: p={p_val:.6f}, effect={effect:.3f}")
    else:
        print("No LIWC columns found. Please check your column names.")