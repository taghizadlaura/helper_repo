import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt

#Load the data from the data folder
def load_data():
    #Get the path of the data
    current_dir = os.getcwd()
    #parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    data_folder = os.path.join(current_dir, "data")

    title_file_name = 'soc-redditHyperlinks-title.tsv'
    body_file_name = 'soc-redditHyperlinks-body.tsv'

    #Open the data
    df_title = pd.read_csv(os.path.join(data_folder, title_file_name), sep='\t')
    df_body = pd.read_csv(os.path.join(data_folder, body_file_name), sep='\t')

   
    return df_title, df_body

def emb_loader(df):
    """
    Load embeddings for subreddits present in the dataframe.
    """
    subreddit_path = "data\web-redditEmbeddings-subreddits.csv"
    emb_sr = pd.read_csv(subreddit_path, header = None)
    
    relevant_subreddits = set(df['SOURCE_SUBREDDIT']).union(set(df['TARGET_SUBREDDIT']))
    emb_sr_filtered = emb_sr[emb_sr[0].isin(relevant_subreddits)].copy()
    subreddits = emb_sr_filtered.iloc[:, 0].values
    X = emb_sr_filtered.iloc[:, 1:].values 
    return subreddits, X


def extend_properties(df, properties_name):
    """
    Extend the PROPERTIES column of the dataframe into columns of the different text properties and LIWC.

    :param df: pandas DataFrame of the Reddit data
    :return: pandas DataFrame with extended properties
    """
    props = df['PROPERTIES'].str.split(',', expand=True)
    props.columns = properties_name
    props = props.apply(pd.to_numeric, errors='coerce')

    return pd.concat([df.drop(columns=['PROPERTIES']), props], axis=1)

def timestamp(df):
    """
    Extend the columns to get different timestamps 

    :param df: pandas DataFrame of the Reddit data
    :return: pandas DataFrame with extended timestamps
    """
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors='coerce')

    df["YEAR_WEEK"] = df["TIMESTAMP"].dt.strftime('%Y-%U')

    all_weeks = pd.DataFrame({'YEAR_WEEK': df['YEAR_WEEK'].sort_values().unique()})   

    df['year'] = df['TIMESTAMP'].dt.year
    df['month'] = df['TIMESTAMP'].dt.month
    df['day'] = df['TIMESTAMP'].dt.day
    df['hour'] = df['TIMESTAMP'].dt.hour
    df['weekday'] = df['TIMESTAMP'].dt.day_name()
    df['season'] = ((df['month']%12 + 3)//3).map({1:'Winter',2:'Spring',3:'Summer',4:'Fall'})
    bins = [0,5,12,18,24]
    labels = ['Night','Morning','Afternoon','Evening']
    df['day_period'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False, include_lowest=True)

    return df 