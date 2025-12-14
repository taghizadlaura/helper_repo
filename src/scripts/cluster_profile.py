import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
import os 

current_dir = os.getcwd()

def radar_plot(cluster_norm, feature_names):
    """
    Builds one radar chart with a dropdown menu to select the cluster.
    """

    cluster_ids = list(cluster_norm.index)

    fig = go.Figure()

    for i, cluster_id in enumerate(cluster_ids):
        values = cluster_norm.loc[cluster_id].values.tolist()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=feature_names,
            fill='toself',
            name=f"Cluster {cluster_id}",
            visible=True if i == 0 else False  # show only first
        ))

    # Dropdown menu to select cluster
    buttons = []
    for i, cluster_id in enumerate(cluster_ids):
        visible_list = [False] * len(cluster_ids)
        visible_list[i] = True

        buttons.append(dict(
            label=f"{cluster_id}",
            method="update",
            args=[{"visible": visible_list},
                  {"title": f"Cluster {cluster_id}: LIWC Profile"}]
        ))

    fig.update_layout(
        title=f"Cluster {cluster_ids[0]}: LIWC Profile",
        polar=dict(radialaxis=dict(visible=False)),
        showlegend=False,
        width=700,
        height=700,
        updatemenus=[dict(
            type="dropdown",
            x=1.15,                
            y=1.05,
            xanchor="right",       
            yanchor="top",
            buttons=buttons,
            direction="down",
            showactive=True
        )]
    )

    name= "cluster_profile.html"
    file_path = os.path.join(current_dir, 'images', name)

    fig.write_html(file_path)

    fig.show()



def cluster_profiles(df, partition_all):

    LIWC_list = [
        "LIWC_Negate", "LIWC_Swear", "LIWC_Social", "LIWC_Family",
        "LIWC_Friends", "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo",
        "LIWC_Anx", "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech", "LIWC_Sexual",
        "LIWC_Relativ", "LIWC_Achiev", "LIWC_Leisure", "LIWC_Money",
        "LIWC_Relig", "LIWC_Death"
    ]

    properties_name = LIWC_list

    # Compute most negative clusters
    cluster_conflict = (
        df[df["LINK_SENTIMENT"] == -1]
            .groupby("cluster_source")
            .size()
            .rename("conflict_score")
    )

    # Compute mean LIWC for each cluster
    cluster_profiles_df = (
        df.groupby("cluster_source")[properties_name]
          .mean()
          .join(cluster_conflict, how="left")
          .fillna({"conflict_score": 0})
    )

    # Select top 5 and bottom 5 
    top5 = cluster_profiles_df.nlargest(10, "conflict_score")


    # Map cluster to subreddits
    cluster_to_subreddits = defaultdict(list)
    for subreddit, cluster in partition_all.items():
        cluster_to_subreddits[cluster].append(subreddit)

    # Normalize for visualization
    cluster_norm = (
        ( top5[properties_name] - top5[properties_name].min())
        / (top5[properties_name].max() -  top5[properties_name].min())
    )

    # Plot the radar 
    radar_plot(cluster_norm, properties_name)

    return cluster_norm, cluster_to_subreddits
