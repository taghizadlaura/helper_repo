import pandas as pd 
from IPython.display import display

#Top subreddits by activity
def top_subreddits_by_activity(df):
    """
    Display the 20 subreddits that send the more hyperlinks

    :param df: pandas DataFrame of the Reddit data
    :return: 20 most active subreddits 
    """
    counts = (
        df['SOURCE_SUBREDDIT']
        .value_counts()
        .head(20)
        .reset_index()
        .rename(columns={'index': 'SOURCE_SUBREDDIT', 'SOURCE_SUBREDDIT': 'count'})
    )
    print(f"Top {20} SOURCE_SUBREDDIT by number of posts:\n")
    display(counts)

#Top subreddits by send negativity
def top_subreddits_by_negativity(df):
    """
    Display the 20 subreddits that send the more negativity

    :param df: pandas DataFrame of the Reddit data
    :return: 20 most negative subreddits 
    """
    grp = df.groupby('SOURCE_SUBREDDIT').agg(posts=('POST_ID','count'),
                                              neg_share=('is_negative','mean')).reset_index().sort_values('neg_share', ascending=False)
    #Take only subreddits with at least 100 posts
    popular = grp[grp['posts'] >= 100].sort_values('neg_share', ascending=False)
    print("\n Print top subreddits that send the most negative share of LINK_SENTIMENT")
    display(popular.head(20))

#Top subreddits by given negativity 
def top_subreddits_by_negativity_target(df):
    """
    Display the 20 subreddits that received more negativity

    :param df: pandas DataFrame of the Reddit data
    :return: 20 most targetted subreddits 
    """
    grp = df.groupby('TARGET_SUBREDDIT').agg(posts=('POST_ID','count'),
                                              neg_share=('is_negative','mean')).reset_index().sort_values('neg_share', ascending=False)
    #Take only subreddits with at least 100 posts
    popular = grp[grp['posts'] >= 100].sort_values('neg_share', ascending=False)
    print("\n Print top subreddits that receive the most negative share of LINK_SENTIMENT")
    display(popular.head(20))

#Top subreddits that exchange the most
def top_exchange(df):
    """
    Display the 20 subreddits that exchange the more

    :param df: pandas DataFrame of the Reddit data
    :return: 20 most paired subreddits 
    """
    grp = df.groupby(['SOURCE_SUBREDDIT','TARGET_SUBREDDIT']).agg(posts=('POST_ID','count'),
                                              neg_share=('is_negative','mean')).reset_index().sort_values('posts', ascending=False)
    popular = grp[grp['posts'] >= 50].sort_values('posts', ascending=False)
    print("\n Print top subreddits that exchange the most")
    display(popular.head(20))

