import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from src.utils.plot_radar import plot_radar

LIWC_list = [
 "LIWC_Negate", "LIWC_Swear", "LIWC_Social", "LIWC_Family",
 "LIWC_Friends", "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo",
 "LIWC_Anx", "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech", "LIWC_Sexual",
 "LIWC_Relativ", "LIWC_Achiev", "LIWC_Leisure", "LIWC_Money",
 "LIWC_Relig", "LIWC_Death"
]

#text_features = ["num_chars","num_chars_no_space","frac_alpha","frac_upper"] 
properties_name = LIWC_list # text_features + 


def cluster_profiles(df, partition_all):
    # Conflict score per cluster
    cluster_conflict = (
        df[df["LINK_SENTIMENT"] == -1]
            .groupby("cluster_source")
            .size()
            .rename("conflict_score")
    )

    # Cluster linguistic profiles
    cluster_profiles = (
        df
            .groupby("cluster_source")[properties_name]
            .mean()
            .join(cluster_conflict, how="left")
            .fillna({"conflict_score": 0})
    )

    # Select 5 highest & 5 lowest conflict clusters

    top5 = cluster_profiles.nlargest(5, "conflict_score")
    bottom5 = cluster_profiles.nsmallest(5, "conflict_score")

    clusters_to_plot = pd.concat([top5, bottom5])

    # Build cluster to subreddit mapping
    cluster_to_subreddits = defaultdict(list)
    for subreddit, cluster in partition_all.items():
        cluster_to_subreddits[cluster].append(subreddit)

    # Normalize features (min-max across all clusters)
    cluster_norm = (clusters_to_plot[properties_name] - clusters_to_plot[properties_name].min()) / \
                (clusters_to_plot[properties_name].max() - clusters_to_plot[properties_name].min())
    for cluster_id in cluster_norm.index:
        conflict = clusters_to_plot.loc[cluster_id, "conflict_score"]
        row = cluster_norm.loc[[cluster_id]]   # keep as 1-row DataFrame
        plot_radar(cluster_id, row, properties_name)

    return cluster_norm, cluster_to_subreddits