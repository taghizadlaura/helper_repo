import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
current_dir = os.getcwd()

def plot_inter_cluster_conflict_matrix(clusters, M_sym, title="Cross-Cluster Conflict Matrix"):
    """
    Interactive Plotly heatmap showing symmetric conflict intensity
    between clusters.
    """

    fig = go.Figure(data=go.Heatmap(
        z=M_sym,
        x=clusters,
        y=clusters,
        colorscale="Reds",
        hoverongaps=False,
        colorbar=dict(title="Conflict Intensity")
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis=dict(title="Target Cluster", tickangle=45),
        yaxis=dict(title="Source Cluster"),
        width=900,
        height=800,
    )

    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Save HTML file
    file_path = os.path.join(current_dir, 'images', 'symetric_matrix.html')
    fig.write_html(
        file_path)

    return fig
