import pandas as pd 
from tqdm import tqdm 
from IPython.display import display


def safe_parse_timestamp(df, col='TIMESTAMP'):

    ts = pd.to_datetime(df[col], errors='coerce')  # pas de format strict
    return ts


def pair_conflict(df, properties_name): 

    df['TIMESTAMP'] = safe_parse_timestamp(df)
    df['hour'] = df['TIMESTAMP'].dt.floor('h')
    df['hour']=pd.to_datetime(df['hour'])
    # Aggregate by pair of source and target 
    df_pair_conflict_hour = df.groupby(['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'hour']).agg(
    **{col: (col, 'mean') for col in properties_name},
    LINK_SENTIMENT=('LINK_SENTIMENT', 'mean'),
    n_links=('POST_ID', 'count')
    ).reset_index()

    #Rename the column 
    df_pair_conflict_hour.rename(columns={
        'SOURCE_SUBREDDIT': 'source',
        'TARGET_SUBREDDIT': 'target'
    }, inplace=True)

    #Sort rows by hour reindexed
    df_pair_conflict_hour = df_pair_conflict_hour.sort_values(by='hour').reset_index(drop=True)

    #Definition of negative events
    neg_pair_conflict_hour = df_pair_conflict_hour[df_pair_conflict_hour['LINK_SENTIMENT'] < 0].copy()
    neg_pair_conflict_hour.rename(columns={'hour':'t0'}, inplace=True)

    ## creation of the df_by_target dictionary
    # it contains the DataFrames of events by target sorted by time
    # IMPORTANT: not only negative events, as it is used to retrieve the target's activities before and after the conflict
    # with t being the name of the target, and g the associated DataFrame
    dict_df_by_target = {t: g.sort_values('hour').set_index('hour') 
                for t, g in df_pair_conflict_hour.groupby('target')}
    # dataframe contains all events that the target received, sorted by time and indexed by hour

    # creation of the df_by_source dictionary
    # it contains the DataFrames of events by source sorted by time
    dict_df_by_source = {s: g.sort_values('hour').set_index('hour') 
                for s, g in df_pair_conflict_hour.groupby('source')}
    return dict_df_by_target, dict_df_by_source, neg_pair_conflict_hour


def delta_LIWC_0(
    dict_df_by_target,
    dict_df_by_source,
    neg_pair_conflict_hour,
    properties_name,
    window_pre=pd.Timedelta(days=30),
    window_segments=[(0, 6), (6, 12), (12, 24), (24, 48)],
    avoid_overlap=True
):
    """
    Build a panel DataFrame with post-conflict LIWC variations.


    Parameters
    ----------
    dict_df_by_target : dict
        Dictionary {target: sub-DataFrame} sorted by timestamp (target activities)
        e.g.: {‘subreddit_A’: DataFrame_A, ‘subreddit_B’: DataFrame_B, ...}
    neg_events_micro : DataFrame
        List of events (columns: target, t0, ... + conflict features)
    properties_cols : list
         List of events (columns: target, t0, ... + conflict features)
    window_pre : pd.Timedelta
       Pre-conflict window size
    window_segments : list of (int, int)
        Disjoint windows in hours, e.g., (0.6, (6.12), ...)
    avoid_overlap : bool
        If True, cuts the post windows before the next conflict of the same target

    Returns
    -------
    panel_disjoint : DataFrame
        One line per conflict and time segment post
    """
    print("not the right function, use delta_LIWC instead")
    # Do not analyze the post-conflict time window if another conflict occurs before the end of this window.
    # Preparation: sort events and calculate next_t0.
    neg_pair_conflict_hour = neg_pair_conflict_hour.sort_values(['target', 't0']).copy()
    # sort_values with target and t0
    neg_pair_conflict_hour['next_t0'] = neg_pair_conflict_hour.groupby('target')['t0'].shift(-1)
    # next_t0 corresponds to the next conflict for the same target
    # shift(-1) to get the next line

    records = []
    

    for idx, row in tqdm(neg_pair_conflict_hour.iterrows(), total=len(neg_pair_conflict_hour), desc="Building panel"):
       #for every negative event:
        target = row['target']
        t0 = row['t0']
        if not isinstance(t0, pd.Timestamp):
            t0 = pd.to_datetime(t0, unit='s', errors='coerce')
        next_t0 = row.get('next_t0', pd.NaT)
        # retrieve the next conflict for the same target
        # pd.NaT if it doesn't exist

        # retrieve the target dataframe which is dataframe of 
        df_target = dict_df_by_target.get(target)
        if df_target is None:
            continue
        # retrieve dataframe where target is source
        df_source = dict_df_by_source.get(target)
        if df_source is None:
            continue

        # Pre-conflict window
        pre_window_incoming = df_target.loc[t0 - window_pre : t0 - pd.Timedelta(seconds=1)]
        pre_window_outgoing = df_source.loc[t0 - window_pre : t0 - pd.Timedelta(seconds=1)]
        if pre_window_outgoing.empty :
            continue
        # Pre-conflict feature averages
        pre_means = pre_window_outgoing[properties_name].mean()

        # Disjointed conflict window
        for start_h, end_h in window_segments:
            # calculation of window boundaries
            w_start = t0 + pd.Timedelta(hours=start_h)
            w_end = t0 + pd.Timedelta(hours=end_h)

            # Option: avoid overlaps between conflicts
            # # if the next conflict occurs before the end of the window, skip this window 
           
            # if avoid_overlap and pd.notnull(next_t0) and w_end > next_t0:
            #     break
            # if the next conflict occurs before the end of the window, keep the window but only outgoing activity before next_t0
            if avoid_overlap and pd.notnull(next_t0) and w_end > next_t0:
                w_end = next_t0 - pd.Timedelta(seconds=1)
    
            
            # definition of the post-conflict window
            post_window_outgoing = df_source.loc[w_start:w_end]
            # if empty, we skip
            if post_window_outgoing.empty:
                continue
            # if either pre or post window has less than 2 observations, we skip
            if len(pre_window_outgoing) < 2 or len(post_window_outgoing) < 2:
                continue
            
            # Savings of the results
            rec = {
                'target': target,
                't0': t0,
                'window_start_h': start_h,
                'window_end_h': end_h
            }

            # Average variation for each feature: post mean - pre mean
            for col in properties_name:
                rec[f'delta_{col}'] = post_window_outgoing[col].mean() - pre_means[col]
                rec[f'incoming_{col}'] = row[col]

            # Event registration
            records.append(rec)

    # Final dataframe construction 
    panel_disjoint = pd.DataFrame(records)


    return panel_disjoint


def delta_LIWC_indiv(
    dict_df_by_target,
    dict_df_by_source,
    neg_pair_conflict_hour,
    properties_name,
    window_pre=pd.Timedelta(days=30),
    window_end = 48,
    avoid_overlap=True
):
    """
    Build a panel DataFrame with post-conflict LIWC variations.


    Parameters
    ----------
    dict_df_by_target : dict
        Dictionary {target: sub-DataFrame} sorted by timestamp (target activities)
        e.g.: {‘subreddit_A’: DataFrame_A, ‘subreddit_B’: DataFrame_B, ...}
    neg_events_micro : DataFrame
        List of events (columns: target, t0, ... + conflict features)
    properties_cols : list
         List of events (columns: target, t0, ... + conflict features)
    window_pre : pd.Timedelta
       Pre-conflict window size
    window_end : int
        End of the post-conflict window in hours
    avoid_overlap : bool
        If True, cuts the post windows before the next conflict of the same target

    Returns
    -------
    panel_disjoint : DataFrame
        One line per conflict and time segment post
    """
    print("function for individual case")
    # Do not analyze the post-conflict time window if another conflict occurs before the end of this window.
    # Preparation: sort events and calculate next_t0.
    neg_pair_conflict_hour = neg_pair_conflict_hour.sort_values(['target', 't0']).copy()
    # sort_values with target and t0
    neg_pair_conflict_hour['next_t0'] = neg_pair_conflict_hour.groupby('target')['t0'].shift(-1)
    # next_t0 corresponds to the next conflict for the same target
    # shift(-1) to get the next line

    records = []
    

    for idx, row in tqdm(neg_pair_conflict_hour.iterrows(), total=len(neg_pair_conflict_hour), desc="Building panel"):
       #for every negative event:
        target = row['target']
        t0 = row['t0']
        if not isinstance(t0, pd.Timestamp):
            t0 = pd.to_datetime(t0, unit='s', errors='coerce')
        next_t0 = row.get('next_t0', pd.NaT)
        # retrieve the next conflict for the same target
        # pd.NaT if it doesn't exist

        # retrieve the target dataframe which is dataframe of 
        df_target = dict_df_by_target.get(target)
        if df_target is None:
            continue
        # retrieve dataframe where target is source
        df_source = dict_df_by_source.get(target)
        if df_source is None:
            continue

        # Pre-conflict window
        pre_window_incoming = df_target.loc[t0 - window_pre : t0 - pd.Timedelta(seconds=1)]
        pre_window_outgoing = df_source.loc[t0 - window_pre : t0 - pd.Timedelta(seconds=1)]
        if pre_window_outgoing.empty :
            continue
        # Pre-conflict feature averages
        pre_means = pre_window_outgoing[properties_name].mean()

        # Disjointed conflict window

        start_h = 0
        end_h = window_end
        # calculation of window boundaries
        w_start = t0 + pd.Timedelta(hours=start_h)
        w_end = t0 + pd.Timedelta(hours=end_h)

        # Option: avoid overlaps between conflicts
        # if the next conflict occurs before the end of the window, keep the window but only outgoing activity before next_t0
        if avoid_overlap and pd.notnull(next_t0) and w_end > next_t0:
            w_end = next_t0 - pd.Timedelta(seconds=1)

        
        # definition of the post-conflict window
        post_window_outgoing = df_source.loc[w_start:w_end]
        # if empty, we skip
        if post_window_outgoing.empty:
            continue
        # if either pre or post window has less than 2 observations, we skip
        if len(pre_window_outgoing) < 2 and len(post_window_outgoing) < 2:
            continue
        
        # Savings of the results
        rec = {
            'source': row['source'],
            'target': target,
            't0': t0,
            'window_start_h': start_h,
            'window_end_h': end_h
        }

        # Average variation for each feature: post mean - pre mean
        for col in properties_name:
            rec[f'delta_{col}'] = post_window_outgoing[col].mean() - pre_means[col]
            rec[f'incoming_{col}'] = row[col]

        # Event registration
        records.append(rec)

    # Final dataframe construction 
    panel_disjoint = pd.DataFrame(records)


    return panel_disjoint


def delta_LIWC(
    dict_df_by_target,
    dict_df_by_source,
    neg_pair_conflict_hour,
    properties_name,
    window_pre=pd.Timedelta(days=30),
    window_end = 48,
    avoid_overlap=True
):
    """
    Build a panel DataFrame with post-conflict LIWC variations.


    Parameters
    ----------
    dict_df_by_target : dict
        Dictionary {target: sub-DataFrame} sorted by timestamp (target activities)
        e.g.: {‘subreddit_A’: DataFrame_A, ‘subreddit_B’: DataFrame_B, ...}
    neg_events_micro : DataFrame
        List of events (columns: target, t0, ... + conflict features)
    properties_cols : list
         List of events (columns: target, t0, ... + conflict features)
    window_pre : pd.Timedelta
       Pre-conflict window size
    window_end : int
        End of the post-conflict window in hours
    avoid_overlap : bool
        If True, cuts the post windows before the next conflict of the same target

    Returns
    -------
    panel_disjoint : DataFrame
        One line per conflict and time segment post
    """
    print("function for general case")
    # Do not analyze the post-conflict time window if another conflict occurs before the end of this window.
    # Preparation: sort events and calculate next_t0.
    neg_pair_conflict_hour = neg_pair_conflict_hour.sort_values(['target', 't0']).copy()
    # sort_values with target and t0
    neg_pair_conflict_hour['next_t0'] = neg_pair_conflict_hour.groupby('target')['t0'].shift(-1)
    # next_t0 corresponds to the next conflict for the same target
    # shift(-1) to get the next line

    records = []
    

    for idx, row in tqdm(neg_pair_conflict_hour.iterrows(), total=len(neg_pair_conflict_hour), desc="Building panel"):
       #for every negative event:
        target = row['target']
        t0 = row['t0']
        if not isinstance(t0, pd.Timestamp):
            t0 = pd.to_datetime(t0, unit='s', errors='coerce')
        next_t0 = row.get('next_t0', pd.NaT)
        # retrieve the next conflict for the same target
        # pd.NaT if it doesn't exist

        # retrieve the target dataframe which is dataframe of 
        df_target = dict_df_by_target.get(target)
        if df_target is None:
            continue
        # retrieve dataframe where target is source
        df_source = dict_df_by_source.get(target)
        if df_source is None:
            continue

        # Pre-conflict window
        pre_window_incoming = df_target.loc[t0 - window_pre : t0 - pd.Timedelta(seconds=1)]
        pre_window_outgoing = df_source.loc[t0 - window_pre : t0 - pd.Timedelta(seconds=1)]
        if pre_window_outgoing.empty :
            continue
        # Pre-conflict feature averages
        pre_means = pre_window_outgoing[properties_name].mean()

        # Disjointed conflict window

        start_h = 0
        end_h = window_end
        # calculation of window boundaries
        w_start = t0 + pd.Timedelta(hours=start_h)
        w_end = t0 + pd.Timedelta(hours=end_h)

        # Option: avoid overlaps between conflicts
        # if the next conflict occurs before the end of the window, keep the window but only outgoing activity before next_t0
        if avoid_overlap and pd.notnull(next_t0) and w_end > next_t0:
            w_end = next_t0 - pd.Timedelta(seconds=1)

        
        # definition of the post-conflict window
        post_window_outgoing = df_source.loc[w_start:w_end]
        # if empty, we skip
        if post_window_outgoing.empty:
            continue
        # if either pre or post window has less than 1 observation, we skip
        if len(pre_window_outgoing) < 1 and len(post_window_outgoing) < 1:
            continue
        
        # Savings of the results
        rec = {
            'source': row['source'],
            'target': target,
            't0': t0,
            'window_start_h': start_h,
            'window_end_h': end_h
        }

        # Average variation for each feature: post mean - pre mean
        for col in properties_name:
            rec[f'delta_{col}'] = post_window_outgoing[col].mean() - pre_means[col]
            rec[f'incoming_{col}'] = row[col]

        # Event registration
        records.append(rec)

    # Final dataframe construction 
    panel_disjoint = pd.DataFrame(records)


    return panel_disjoint

       
def visualize_results(dict_df_by_target, dict_df_by_source, neg_pair_conflict_hour, properties_name):
    #this function will build a panel DataFrame with post-conflict LIWC variations
    df_delta_LIWC = delta_LIWC(
        dict_df_by_target=dict_df_by_target,
        dict_df_by_source=dict_df_by_source,
        neg_pair_conflict_hour=neg_pair_conflict_hour,
        properties_name=properties_name,
        window_pre=pd.Timedelta(days=30),
        window_end = 48,
        #window_segments=[(0,6), (6,12), (12,24), (24,48)],
        avoid_overlap=True
    )

    # sort by t0 window_start_h target
    df_delta_LIWC = df_delta_LIWC.sort_values(by=['t0','target', 'window_start_h'])

    display(df_delta_LIWC)

    return df_delta_LIWC
    
def visualize_results_indiv(dict_df_by_target, dict_df_by_source, neg_pair_conflict_hour, properties_name):
    #this function will build a panel DataFrame with post-conflict LIWC variations
    df_delta_LIWC = delta_LIWC_indiv(
        dict_df_by_target=dict_df_by_target,
        dict_df_by_source=dict_df_by_source,
        neg_pair_conflict_hour=neg_pair_conflict_hour,
        properties_name=properties_name,
        window_pre=pd.Timedelta(days=30),
        window_end = 48,
        #window_segments=[(0,6), (6,12), (12,24), (24,48)],
        avoid_overlap=True
    )

    # sort by t0 window_start_h target
    df_delta_LIWC = df_delta_LIWC.sort_values(by=['t0','target', 'window_start_h'])

    display(df_delta_LIWC)

    return df_delta_LIWC