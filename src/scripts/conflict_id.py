import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks

TOP_N = int(1)
LIWC_list = [
    "LIWC_Funct", "LIWC_Pronoun", "LIWC_Ppron", "LIWC_I",
    "LIWC_We", "LIWC_You", "LIWC_SheHe", "LIWC_They", "LIWC_Ipron",
    "LIWC_Article", "LIWC_Verbs", "LIWC_AuxVb", "LIWC_Past", "LIWC_Present",
    "LIWC_Future", "LIWC_Adverbs", "LIWC_Prep", "LIWC_Conj", "LIWC_Negate",
    "LIWC_Quant", "LIWC_Numbers", "LIWC_Swear", "LIWC_Social", "LIWC_Family",
    "LIWC_Friends", "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo",
    "LIWC_Anx", "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech", "LIWC_Insight",
    "LIWC_Cause", "LIWC_Discrep", "LIWC_Tentat", "LIWC_Certain", "LIWC_Inhib",
    "LIWC_Incl", "LIWC_Excl", "LIWC_Percept", "LIWC_See", "LIWC_Hear",
    "LIWC_Feel", "LIWC_Bio", "LIWC_Body", "LIWC_Health", "LIWC_Sexual",
    "LIWC_Ingest", "LIWC_Relativ", "LIWC_Motion", "LIWC_Space", "LIWC_Time",
    "LIWC_Work", "LIWC_Achiev", "LIWC_Leisure", "LIWC_Home", "LIWC_Money",
    "LIWC_Relig", "LIWC_Death", "LIWC_Assent", "LIWC_Dissent", "LIWC_Nonflu",
    "LIWC_Filler"
]

def get_top_N_ls(df, group, N=TOP_N):
    """
    Get the top N sources or targets with the most negative LINK_SENTIMENT.

    :param df: pandas DataFrame of the Reddit data
    :param group: 'SOURCE_SUBREDDIT' or 'TARGET_SUBREDDIT' to group by
    :return: pandas Series of top N sources or targets with counts of negative LINK_SENTIMENT
    """
    df_neg = df[df['LINK_SENTIMENT'] == -1]
    top_N = df_neg.groupby(group).size().reset_index(name='count').sort_values(by='count', ascending=False, ignore_index=True).head(N)
    
    return top_N


def plot_weekly_activity(subreddit, df, group):
    """
    Plot weekly activity for a specific subreddit.

    :param subreddit: subreddit name
    :param df: pandas DataFrame of the Reddit data
    :param group: 'SOURCE_SUBREDDIT' or 'TARGET_SUBREDDIT' to group by
    """
    all_weeks = pd.DataFrame({'YEAR_WEEK': df['YEAR_WEEK'].sort_values().unique()})
    # Filter dataframe for the specific subreddit and compute weekly activity
    df_subreddit = df[df[group].str.lower() == subreddit.lower()].copy()
    df_weekly = df_subreddit.groupby('YEAR_WEEK').agg(neg_mentions=('LINK_SENTIMENT', lambda x: (x < 0).sum())).fillna(0)
    df_weekly = all_weeks.merge(df_weekly, on='YEAR_WEEK', how='left').fillna(0)
    df_weekly = df_weekly.sort_values(by='YEAR_WEEK').reset_index(drop=True)

    # Detect peaks in activity
    peaks, properties = find_peaks(df_weekly['neg_mentions'].values, prominence=8, height=10)
    df_weekly['conflict_flag'] = False
    df_weekly.loc[peaks, 'conflict_flag'] = True

    tick_spacing = 4
    plt.figure(figsize=(20,3))
    plt.plot(df_weekly['YEAR_WEEK'].astype(str), df_weekly['neg_mentions'], marker='o')
    plt.title(f'{group.replace("_", " ").title()}: {subreddit} - Weekly Activity')
    plt.xlabel('Year-Week')
    plt.ylabel(f'Weekly of {"Outgoing" if group == "SOURCE_SUBREDDIT" else "Incoming"} Negative LINK_SENTIMENT Posts')
    plt.xticks(ticks=range(0, len(all_weeks), tick_spacing), labels=df_weekly['YEAR_WEEK'].iloc[::tick_spacing].astype(str), rotation=90)

    for i in df_weekly[df_weekly['conflict_flag']].index:
        plt.axvline(x=i, color='r', linestyle='--')
    plt.show()  


def plot_liwc_attack_evolution(subreddit, df, group):
    """
    Plot the evolution of LIWC properties for attacks on a specific subreddit.
    
    :param subreddit: subreddit name
    :param df: pandas DataFrame of the Reddit data
    :param group: 'SOURCE_SUBREDDIT' or 'TARGET_SUBREDDIT' to group by
    """
   
    neg_df = df[
    (df["TARGET_SUBREDDIT"].str.lower() == subreddit.lower()) &
    (df["LINK_SENTIMENT"] < 0)
    ]

    liwc_over_time = (
    neg_df.groupby('YEAR_WEEK')[LIWC_list].mean()
    .rolling(window=4, min_periods=1)
    .mean()
    )

    liwc_over_time[["LIWC_Anger", "LIWC_Swear", "LIWC_Negemo"]].plot(figsize=(12,5))
    plt.title(f"Evolution of linguistic tone in target / source on {subreddit}")
    plt.xlabel("Week")
    plt.ylabel("Average LIWC score")
    plt.show

def plot_liwc_neg_source_evolution(subreddit, df, group):
    """
    Plot the evolution of LIWC properties for negative source posts of a specific subreddit.

    :param subreddit: subreddit name
    :param df: pandas DataFrame of the Reddit data
    :param group: 'SOURCE_SUBREDDIT' or 'TARGET_SUBREDDIT' to group by
    """

    neg_df_source = df[
    (df["SOURCE_SUBREDDIT"].str.lower() == subreddit.lower()) &
    (df["LINK_SENTIMENT"] < 0)
    ]

    liwc_source_over_time = (
    neg_df_source.groupby('YEAR_WEEK')[LIWC_list].mean()
    .rolling(window=4, min_periods=1)
    .mean()
    )

    liwc_source_over_time[["LIWC_Anger", "LIWC_Swear", "LIWC_Negemo"]].plot(figsize=(12,5))
    plt.title(f"Evolution of linguistic tone in negative source from {subreddit}")
    plt.xlabel("Week")
    plt.ylabel("Average LIWC score")
    plt.show()

def show_weekly_activity(df, df_top_N, group):
    """"
    Show weekly activity plots for each of the top N sources or targets with most negative LINK_SENTIMENT.

    :param df: pandas DataFrame of the Reddit data
    :param group: 'SOURCE_SUBREDDIT' or 'TARGET_SUBREDDIT' to group by
    """

    for subreddit in df_top_N[group]:
        plot_weekly_activity(subreddit, df, group)
        plot_liwc_attack_evolution(subreddit, df, group)
        plot_liwc_neg_source_evolution(subreddit, df, group)