# statistical tests on delta of all LIWC features :
# are the deltas significantly different from zero ? (delta = post-pre)
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd 
from tqdm import tqdm 
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

def t_test_delta_LIWC(df_delta_LIWC, properties_name):

    # separate cases by window_start_h
    significant_features = []
    non_significant_features = []
    means_list = []
    p_values_list = []
    feature_names_list = []

    # for each post conflict window 
    for window_start_h in df_delta_LIWC['window_start_h'].unique():
        subset = df_delta_LIWC[df_delta_LIWC['window_start_h'] == window_start_h]
        print(f"\nWindow start hour: {window_start_h}")
        significant_feature = []
        non_significant_feature = []
        means = []
        p_values = []
        feature_names = []

        # for each feature
        for col in properties_name:
            # get non zero deltas
            deltas = subset[f'delta_{col}'].dropna()
            if len(deltas) > 1:
                # t-test between the mean of the provided deltas and 0
                t_stat, p_value = stats.ttest_1samp(deltas, 0)
                means.append(deltas.mean())
                p_values.append(p_value)
                feature_names.append(col)
                if p_value < 0.05:
                    significant_feature.append((col, p_value))
                else:
                    non_significant_feature.append((col, p_value))

        # plot significant and non-significant features for each window_start_h
        means_list.append(means)
        p_values_list.append(p_values)
        feature_names_list.append(feature_names)
        significant_features.append((window_start_h, significant_feature))
        non_significant_features.append((window_start_h, non_significant_feature))
        liwc_features = [f for f in feature_names if f.startswith('LIWC_')]
        non_liwc_features = [f for f in feature_names if f not in liwc_features]
        # plot LIWC features and non-LIWC features separately
        for feature_set, title in [(liwc_features, 'LIWC Features'), (non_liwc_features, 'Non-LIWC Features')]:
            indices = [feature_names.index(f) for f in feature_set]
            means_subset = [means[i] for i in indices]
            p_values_subset = [p_values[i] for i in indices]
            x = range(len(feature_set))

            plt.figure(figsize=(12, 6))
            bars = plt.bar(x, means_subset, color=['red' if p_values_subset[i] < 0.05 else 'blue' for i in range(len(p_values_subset))])
            plt.xticks(x, feature_set, rotation='vertical')
            plt.xlabel('Features')
            plt.ylabel('Mean Delta')
            plt.legend(handles=[
                plt.Line2D([0], [0], color='red', lw=4, label='Significant (p < 0.05)'),
                plt.Line2D([0], [0], color='blue', lw=4, label='Non-Significant (p >= 0.05)')
            ])
            plt.title(f'Mean Delta of {title} (Window Start Hour: {window_start_h})')
            plt.axhline(0, color='black', linewidth=0.8)
            plt.tight_layout()
            plt.show()
        

    return significant_features, non_significant_features, means_list, p_values_list, feature_names_list

def delta_LIWC_timeseries(
    dict_df_by_target,
    dict_df_by_source,
    neg_pair_conflict_hour,
    properties_name,
    window_pre=pd.Timedelta(days=30),
    window_segments=[(0, 6), (6, 12), (12, 24), (24, 48)],
    avoid_overlap=True
):
    """
    Build a panel DataFrame with post-conflict LIWC variations across multiple time segments.
    """
    print("Building LIWC timeseries with multiple segments")
    
    # Sort events and calculate next_t0
    neg_pair_conflict_hour = neg_pair_conflict_hour.sort_values(['target', 't0']).copy()
    neg_pair_conflict_hour['next_t0'] = neg_pair_conflict_hour.groupby('target')['t0'].shift(-1)

    records = []
    
    for idx, row in tqdm(neg_pair_conflict_hour.iterrows(), total=len(neg_pair_conflict_hour), desc="Building panel"):
        target = row['target']
        t0 = row['t0']
        if not isinstance(t0, pd.Timestamp):
            t0 = pd.to_datetime(t0, unit='s', errors='coerce')
        next_t0 = row.get('next_t0', pd.NaT)

        # Retrieve dataframes
        df_target = dict_df_by_target.get(target)
        df_source = dict_df_by_source.get(target)
        if df_target is None or df_source is None:
            continue

        # Pre-conflict window (same for all segments)
        pre_window_outgoing = df_source.loc[t0 - window_pre : t0 - pd.Timedelta(seconds=1)]
        if pre_window_outgoing.empty:
            continue
            
        pre_means = pre_window_outgoing[properties_name].mean()

        # Process each time segment
        for start_h, end_h in window_segments:
            # Calculate window boundaries
            w_start = t0 + pd.Timedelta(hours=start_h)
            w_end = t0 + pd.Timedelta(hours=end_h)

            # Avoid overlaps with next conflict
            if avoid_overlap and pd.notnull(next_t0) and w_end > next_t0:
                w_end = next_t0 - pd.Timedelta(seconds=1)
                if (w_end - w_start).total_seconds() < 3600:
                    continue

            # Post-conflict window for this segment
            post_window_outgoing = df_source.loc[w_start:w_end]
            if post_window_outgoing.empty:
                continue

            # Calculate deltas for this time segment
            rec = {
                'source': row['source'],
                'target': target,
                't0': t0,
                'window_start_h': start_h,
                'window_end_h': end_h,
                'segment_label': f"{start_h}-{end_h}h"
            }

            # Average variation for each feature: post mean - pre mean
            for col in properties_name:
                rec[f'delta_{col}'] = post_window_outgoing[col].mean() - pre_means[col]
                rec[f'incoming_{col}'] = row[col]  

            records.append(rec)

    panel_timeseries = pd.DataFrame(records)
    return panel_timeseries