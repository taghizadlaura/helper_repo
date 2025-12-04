# Test causality with a Granget test. A low p-value (< 0.05) suggests that one series Granger-causes the other. 
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

import sys, os
import plotly.express as px


# Supress output of grangercausalitytests because Verbose doesn't work properly
def granger_silent(data, maxlag=5):
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        result = grangercausalitytests(data, maxlag=maxlag)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    return result

# Aggregate time series per subreddit
def create_time_series(subreddit, df, freq='D'):
    # Attacks received: negative posts targeting this subreddit
    attacked = (
        df[(df['TARGET_SUBREDDIT']==subreddit) & (df['LINK_SENTIMENT']==-1)]
        .groupby(pd.Grouper(key='TIMESTAMP', freq=freq))
        .size()
        .rename('attacked')
    )
    
    # Negative behavior: negative posts sent from this subreddit
    negative_behavior = (
        df[(df['SOURCE_SUBREDDIT']==subreddit) & (df['LINK_SENTIMENT']==-1)]
        .groupby(pd.Grouper(key='TIMESTAMP', freq=freq))
        .size()
        .rename('negative_behavior')
    )
    
    # Combine into single DataFrame
    ts = pd.concat([negative_behavior, attacked], axis=1).fillna(0)
    
    # Ensure continuous date index
    idx = pd.date_range(ts.index.min(), ts.index.max(), freq=freq)
    ts = ts.reindex(idx, fill_value=0)
    
    return ts

# Check stationarity and difference if needed to apply Granger test
def make_stationary(ts):
    stationary = pd.DataFrame()
    for col in ts.columns:
        result = adfuller(ts[col])
        if result[1] > 0.05:  # non-stationary
            stationary[col] = ts[col].diff().dropna()
        else:
            stationary[col] = ts[col]
    stationary = stationary.dropna()
    return stationary

# Run Granger causality test
def test_granger(ts, maxlag=5):
    """
    ts: DataFrame with columns ['negative_behavior', 'attacked']
    """
    # [] to have only this lag
    results = granger_silent(ts[['negative_behavior', 'attacked']], maxlag=maxlag)
    return results


def granger_analysis(df):
    #Get the 20 subreddits that have the highest number of received negative sentiment
    top_subreddits = (
        df[df['LINK_SENTIMENT'] == -1]
        .groupby('TARGET_SUBREDDIT')
        .size()
        .sort_values(ascending=False)
        .head(20)
        .index
        .tolist()
    )
    records=[]

    # Run Granger causality tests for top subreddits
    for subreddit in top_subreddits:
        for freq in ['H','D']:  
            #print(f"\nGranger causality results for subreddit: {subreddit} at frequency: {freq}")
            ts_raw=create_time_series(subreddit, df, freq=freq)
            # Error x is constant → skip
            if ts_raw['attacked'].nunique() <= 1 or ts_raw['negative_behavior'].nunique() <= 1:
                print(f"For subreddit '{subreddit}' at frequency '{freq}':")
                print("Time series is constant; skipping Granger test.")
                continue
            ts = make_stationary(ts_raw)
            # Lag indicates how many time steps back we look 
            granger_results = test_granger(ts, maxlag=5)
            # Interpret results for lags
            for lag in range(1, 6):
                f_test = granger_results[lag][0]['ssr_ftest']
                f_stat, p_value, df_denom, df_num = f_test
                #print(f"Lag {lag}: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")
                # Interpretation    
                if  p_value < 0.05:
                    #continue
                    print(f"For subreddit '{subreddit}' at frequency '{freq}':")
                    print(f"At lag {lag}, we reject the null hypothesis: being attacked Granger-causes negative behavior.")

                records.append({
                    'subreddit': subreddit,
                    'frequency': freq,
                    'lag': lag,
                    'p_value': p_value,
                    'F_stat': f_stat
                })
    return records

                