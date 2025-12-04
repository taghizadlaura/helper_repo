import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

def plot_num_cluster_time(x):
    plt.figure(figsize=(12, 6))
    plt.plot(list(x.keys()), list(x.values()), marker='o')
    plt.title('Evolution of Number of Communities Over Time')
    plt.xlabel('Period (Month)')
    plt.ylabel('Number of Communities')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


import plotly.express as px

def plot_interactive(subreddits, clusters, X_embedded, title):
    """
    Visualisation interactive des clusters KMeans avec t-SNE.
    
    Parameters:
    -----------
    subreddits : list or pd.Series
        Name of subreddits
    clusters : np.ndarray
        Labels of the clusters
    X_embedded : np.ndarray
        Coordinates 2D t-SNE
    title : str
    """
    import pandas as pd
    df_plot = pd.DataFrame({
        "subreddit": subreddits,
        "cluster": clusters,
        "x": X_embedded[:, 0],
        "y": X_embedded[:, 1]
    })
    df_plot["cluster_str"] = df_plot["cluster"].astype(str)

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="cluster_str",
        hover_name="subreddit",
        hover_data={"cluster": True, "x": False, "y": False, "cluster_str": False},
        title=title,
        height=700
    )

    fig.update_traces(marker=dict(size=6), selector=dict(mode="markers"))
    fig.update_layout(
        legend_title_text="KMeans cluster (click to toggle)",
        legend=dict(itemsizing="constant")
    )
    fig.show()

def plot_cluster_overlap(df_confusion, title="Heat Map of confusion Matrix of Cluster Overlap"):
    """
    Show heatmap of confusion matrix between two clustering methods.
    """
    plt.figure(figsize=(10,8))
    sns.heatmap(df_confusion, annot=True, fmt="d", cmap="viridis")
    plt.title(title)
    plt.xlabel("Cluster method 1")
    plt.ylabel("Cluster method 2")
    plt.show()