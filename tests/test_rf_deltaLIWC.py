from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm


def rf_delta_LIWC_analysis(df_delta_LIWC, properties_name, window=None, min_samples=5, r2_threshold=0.05):
    """
    Perform Random Forest regression to analyze the influence of incoming LIWC features
    on the delta LIWC features post-conflict.


    Parameters:
    - df_delta_LIWC: DataFrame containing delta LIWC features and incoming features.
    - properties_name: List of LIWC property names to analyze.

    Returns:
    - df_rf_summary: Summary DataFrame of top features per delta.
    - df_rf_full_importances: Detailed feature importances DataFrame.
    - df_importance_matrix: Pivot table for heatmap visualization.
    """

    # By default, analyze all windows. To restrict to a specific window, pass
    # `window=(start_h, end_h)`.
    if window is None:
        df_rf_data = df_delta_LIWC.copy()
    else:
        start_h, end_h = window
        df_rf_data = df_delta_LIWC[
            (df_delta_LIWC['window_start_h'] == start_h) &
            (df_delta_LIWC['window_end_h'] == end_h)
        ].copy()

    # incoming features 
    requested_incoming = [f'incoming_{f}' for f in properties_name]
    incoming_cols = [c for c in requested_incoming if c in df_rf_data.columns]

    # storage
    summary = [] # summary of top features per delta 
    feature_importance_records = [] # detailed feature importances
    skip_logs = []
    total_features = 0
    trained_models = 0

    # Random Forest parameters
    rf_params = {
        'n_estimators': 200, # number of trees : more trees for better stability
        'max_depth': 15, # maximum depth of each tree : control overfitting
        'min_samples_leaf': 5,  # minimum samples per leaf to avoid overfitting
        'random_state': 42, # for reproducibility
        'n_jobs': -1 # use all available cores
    }


    # Train RF per delta feature
    for feature in tqdm(properties_name, desc="Training Random Forests"):
        feature_col = f'delta_{feature}'
        total_features += 1

        
        # skip if this delta column is not present
        if feature_col not in df_rf_data.columns:
            msg = f"Skipping {feature}: delta column {feature_col} not present"
            #print(msg)
            skip_logs.append({'feature': feature, 'reason': 'missing_delta', 'detail': feature_col})
            continue

        # select only present predictors and drop rows with NaNs in predictors/target
        cols_here = incoming_cols + [feature_col]
        data = df_rf_data[cols_here].dropna()
        if data.shape[0] < min_samples:
            msg = f"Skipping {feature}: only {data.shape[0]} rows (< min_samples={min_samples})"
            #print(msg)
            skip_logs.append({'feature': feature, 'reason': 'insufficient_rows', 'n_rows': int(data.shape[0])})
            continue

        # predictors and target
        X_df = data[incoming_cols].copy()
        y = data[feature_col].values  # shape (n_samples,)

        # drop constant predictors (zero variance)
        nunique = X_df.nunique(dropna=True)
        keep_cols = list(nunique[nunique > 1].index)
        if len(keep_cols) == 0:
            msg = f"Skipping {feature}: no usable (varying) incoming predictors"
            #print(msg)
            skip_logs.append({'feature': feature, 'reason': 'no_varying_predictors', 'incoming_count': int(len(incoming_cols))})
            continue
        if len(keep_cols) < len(incoming_cols):
            # update X to keep only varying predictors
            X_df = X_df[keep_cols]

        # skip if target is constant
        if pd.Series(y).nunique() <= 1:
            msg = f"Skipping {feature}: target delta is constant (n_unique={pd.Series(y).nunique()})"
            #print(msg)
            skip_logs.append({'feature': feature, 'reason': 'constant_target', 'n_unique': int(pd.Series(y).nunique())})
            continue

        X = X_df.values  # shape (n_conflicts, n_features)

        # standardize X and y (safe: y has variance as checked above)
        X_scaled = StandardScaler().fit_transform(X)
        y_scaled = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()  # flatten for RF

        # train/test split - ensure reasonable test set size
        n_samples = X_scaled.shape[0]
        test_size = min(0.4, max(0.2, 30.0 / float(n_samples)))
        # guard: train_test_split requires 1 <= n_test < n_samples
        n_test = int(max(1, round(test_size * n_samples)))
        if n_test >= n_samples:
            # fallback to a safer split
            test_size = 0.2

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, test_size=test_size, random_state=42
            )
        except Exception:
            msg = f"Skipping {feature}: train_test_split failed"
            #print(msg)
            skip_logs.append({'feature': feature, 'reason': 'split_failed'})
            continue

        # fit Random Forest with basic exception handling
        try:
            rf = RandomForestRegressor(**rf_params)
            rf.fit(X_train, y_train)
        except Exception:
            msg = f"Skipping {feature}: RF training failed"
            #print(msg)
            skip_logs.append({'feature': feature, 'reason': 'training_failed'})
            continue

        # compute R2 (guard against errors)
        try:
            r2 = rf.score(X_test, y_test)
        except Exception:
            msg = f"Skipping {feature}: scoring failed"
            #print(msg)
            skip_logs.append({'feature': feature, 'reason': 'scoring_failed'})
            continue
        if r2 is None or (isinstance(r2, float) and r2 < r2_threshold):
            msg = f"Skipping {feature}: low r2 ({r2:.4f}) < threshold {r2_threshold}"
            #print(msg)
            skip_logs.append({'feature': feature, 'reason': 'low_r2', 'r2': float(r2) if r2 is not None else None})
            continue  # skip uninformative deltas

        # store feature importances; index corresponds to keep_cols
        importances = pd.Series(rf.feature_importances_, index=keep_cols).sort_values(ascending=False)
        for inc_feat, imp_val in importances.items():
            feature_importance_records.append({
                'target_delta': feature, # the current studied feature
                'incoming_feature': inc_feat.replace('incoming_', ''), # the incoming feature name
                'importance': imp_val, # importance value
                'r2_test': r2 # r2 value
            })

        # summary: top 3 features (can be adjusted)
        top_features = ", ".join(importances.index[:3].str.replace('incoming_', '')) # the top 3 feature names
        top_importances = ", ".join([f"{v:.3f}" for v in importances.values[:3]]) # their importance values
        summary.append({
            'target_delta': feature, # the current studied feature
            'top_3_incoming_features': top_features,
            'top_3_importances': top_importances,
            'r2_test': r2
        })
        trained_models += 1


    # Create summary DataFrames
    if len(summary) == 0:
        df_rf_summary = pd.DataFrame(
            columns=['target_delta', 'top_3_incoming_features', 'top_3_importances', 'r2_test']
        )
    else:
        df_rf_summary = pd.DataFrame(summary).sort_values('r2_test', ascending=False).reset_index(drop=True)

    if len(feature_importance_records) == 0:
        df_rf_full_importances = pd.DataFrame(
            columns=['target_delta', 'incoming_feature', 'importance', 'r2_test']
        )
        df_importance_matrix = pd.DataFrame()
    else:
        df_rf_full_importances = pd.DataFrame(feature_importance_records)
        # heatmap (rows: target_delta, columns: incoming_feature, values: importance)
        df_importance_matrix = df_rf_full_importances.pivot_table(
            index='target_delta', columns='incoming_feature', values='importance', fill_value=0
        )

    if window is None:
        window_desc = 'all windows'
    else:
        window_desc = f'{window[0]}-{window[1]}h post-conflict'
    print(f"\n==== Random Forest Summary ({window_desc}) ====")
    print(df_rf_summary.head(15))
    print(f"\nProcessed features: {total_features}; Trained models: {trained_models}; Skipped: {total_features - trained_models}")
    if len(skip_logs) > 0:
        print('\nExample skip logs:')
        import json
        print(json.dumps(skip_logs[:20], indent=2))
    return df_rf_summary, df_rf_full_importances, df_importance_matrix