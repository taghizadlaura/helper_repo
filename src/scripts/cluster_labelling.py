import pandas as pd

def label_cluster(df):
    # Open the cluster and label the 20 most virulent one it by hand 
    df_clusters = pd.read_csv('cluster_full_data_2.csv')

    subreddit_to_theme = {}

    for _, row in df_clusters.iterrows():
        themes = row["Cluster_ID"]
        
        # Get all subreddits per theme
        subs = str(row["All_Members"]).split(",")
        subs = [s.strip().lower() for s in subs]

        for s in subs:
            if s:  # ignore empty strings
                subreddit_to_theme[s] = themes


    # Map cluster assignement
    df["cluster_source"] = df["SOURCE_SUBREDDIT"].str.lower().map(subreddit_to_theme)
    df["cluster_target"] = df["TARGET_SUBREDDIT"].str.lower().map(subreddit_to_theme)


    cluster_conflict = (
            df[df["LINK_SENTIMENT"] == -1]
                .groupby("cluster_source")
                .size()
                .rename("conflict_score")
    )

    df_clusters["conflict_score"] = df_clusters["Cluster_ID"].map(cluster_conflict)


    top20 = df_clusters.nlargest(20, "conflict_score")

    display(top20)

    auto_labels = [
        "Drama", "Politics", "Everyday life", "Animals", "Religion",
        "Relantionships", "General knowledge", "Conspiracy", "Gaming",
        "Fitness", "Language", "Television", "Complain about reddit",
        "Cryptocurrency", "Country", "Canada", "Music", "Legal"
    ]

    cluster_to_theme = dict(enumerate(auto_labels))

    # Assign theme to clusters
    df_clusters["theme"] = df_clusters["Cluster_ID"].map(cluster_to_theme)


    subreddit_to_theme = {}

    for _, row in df_clusters.iterrows():
        themes = row["theme"]
        
        # Get all subreddits per theme
        subs = str(row["All_Members"]).split(",")
        subs = [s.strip().lower() for s in subs]

        for s in subs:
            if s:  # ignore empty strings
                subreddit_to_theme[s] = themes

    # Map cluster assignement
    df["cluster_source"] = df["SOURCE_SUBREDDIT"].str.lower().map(subreddit_to_theme)
    df["cluster_target"] = df["TARGET_SUBREDDIT"].str.lower().map(subreddit_to_theme)



    return subreddit_to_theme