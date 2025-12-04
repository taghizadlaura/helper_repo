import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from scipy.stats import skew, pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


plot_examples=["LIWC_Anger", "LIWC_Negemo", "LIWC_Posemo"]

def check_skewness(df, properties_name ):
    """
    Check the distribution of the data

    :param df: pandas DataFrame of the Reddit data
    :param properties_name: all linguistic characteristics that we want to study
    """

    df_log = df.copy()

    # Calculate skewness for each property.
    skewness = df[properties_name].apply(lambda x: skew(x.dropna()))
    print("\n=== Variables skewness===")
    display(skewness.sort_values(key=abs, ascending=False))

    # Identify features with high skewness (absolute value > 1)
    skewed_feats = skewness[abs(skewness) >1 ].index
    print(f"\n{len(skewed_feats)} Variables with skewness > {1} :")
    print(f"\n Exemple of histogram")

    # Visualize original vs log-transformed distributions for example features
    for f in plot_examples:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

      
        sns.histplot(df[f], kde=True, bins=40, ax=axes[0])
        axes[0].set_title(f"{f} - Original", fontsize=12)
        axes[0].set_xlabel(f)
        axes[0].set_ylabel("Fréquence")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        lower = 0
        upper = df[f].quantile(0.99)
        axes[0].set_xlim(left=lower, right=upper)

        if (df[f] <= 0).any():
            shift = abs(df[f].min()) + 1
            transformed = np.log1p(df[f] + shift)
        else:
            transformed = np.log1p(df[f])

        sns.histplot(transformed, kde=True, bins=40, ax=axes[1], color="orange")
        axes[1].set_title(f"{f} - Log1p", fontsize=12)
        axes[1].set_xlabel(f"Log1p({f})")
        axes[1].set_ylabel("Fréquence")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        lower_t = 0
        upper_t = transformed.quantile(0.99)
        axes[1].set_xlim(left=lower_t, right=upper_t)

        plt.tight_layout()
        plt.show()

def naive_analysis(df, properties_name):
    """
    Make a naive analysis of the distribution of the properties

    :param df: pandas DataFrame of the Reddit data
    :param properties_name: all linguistic characteristics that we want to study
    """
    # Create binary indicator for negative sentiment
    df["is_negative"] = (df['LINK_SENTIMENT'] == -1).astype(int)
    df_liwc = pd.concat([df[properties_name], df["LINK_SENTIMENT"], df["is_negative"]], axis=1)

    # Calculate average LIWC features by sentiment category
    avg_liwc_by_sentiment = df_liwc.groupby('LINK_SENTIMENT')[properties_name[1:]].mean()
    print("\nAverage LIWC/Text features by LINK_SENTIMENT:")
    display(avg_liwc_by_sentiment)

    # Boxplots for key features across different sentiment categories
    print("\n PLot boxplot of key LIWC features in function of hyperlink sentiment")
    key_features = ['LIWC_Negemo', 'LIWC_Anger', 'LIWC_Sad', 'LIWC_Posemo']
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey=True, sharex=True)
    for i, feature in enumerate(key_features):
        row = i // 2
        col = i % 2
        sns.boxplot(x='LINK_SENTIMENT', y=feature, data=df, ax=axs[row, col])
        axs[row, col].set_title(f"{feature} by LINK_SENTIMENT")
        axs[row, col].set_xlabel("LINK_SENTIMENT")
        axs[row, col].set_ylabel(feature)
    plt.tight_layout()
    plt.show()

    # Compare correlation patterns between negative vs neutral/positive posts
    print("\n Print the correlation heatmaps between LIWC and text features in function of hyperlink sentiment")
    corr_neg = df_liwc[df_liwc['LINK_SENTIMENT']==-1][properties_name[1:]].corr()
    corr_pos = df_liwc[df_liwc['LINK_SENTIMENT']!=-1][properties_name[1:]].corr()
    fig, ax = plt.subplots(1, 2, figsize=(14,6))
    sns.heatmap(corr_neg, ax=ax[0], cmap='coolwarm', center=0)
    sns.heatmap(corr_pos, ax=ax[1], cmap='coolwarm', center=0)
    ax[0].set_title('Negative Posts Correlations')
    ax[1].set_title('Neutral/Positive Posts Correlations')
    plt.show()
    
    # Analyze correlation between LIWC features and negativity binary indicator
    print("\n  Print the correlation heatmap between LIWC and text features and negativity of the hyperlink")
    
    corr_with_neg = df_liwc[properties_name].apply(lambda x: x.corr(df_liwc['is_negative']))
    corr_df = corr_with_neg.to_frame(name='Correlation').sort_values('Correlation', ascending=False)

    plt.figure(figsize=(6, max(6, len(properties_name)*0.3)))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation between LIWC Features and Negativity")
    plt.xlabel("Correlation")
    plt.ylabel("LIWC Feature")
    plt.show()

def statistical_analysis(df, text_features, LIWC_list):
    """
    Make statistical test on features in function of negativity

    :param df: pandas DataFrame of the Reddit data
    :param properties_name: all linguistic characteristics that we want to study
    :param text_features: all text features
    :param all LIWC features
    """
    results = []
    
    # Calculate Pearson correlations between text features and negativity
    for f in text_features:
        if f == 'is_negative':
            continue  
        valid_idx = df[f].notna()
        r, p = pearsonr(df.loc[valid_idx, f], df.loc[valid_idx, 'is_negative'])
        results.append({"Feature": f, "Pearson_r": r, "p_value": p})

    df_corr = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)

    print("\n Top 3 positively correlated text features with negative posts:")
    display(df_corr.sort_values("Pearson_r", ascending=False).head(3)[["Feature", "Pearson_r", "p_value"]])

    print("\n Top 3 negatively correlated text features with negative posts:")
    display(df_corr.sort_values("Pearson_r", ascending=True).head(3)[["Feature", "Pearson_r", "p_value"]])

    results = []
    # Calculate Pearson correlations between LIWC features and negativity
    for f in LIWC_list:
        if f == 'is_negative':
            continue  
        valid_idx = df[f].notna()
        r, p = pearsonr(df.loc[valid_idx, f], df.loc[valid_idx, 'is_negative'])
        results.append({"Feature": f, "Pearson_r": r, "p_value": p})

    df_corr = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)

    print("\n Top 3 positively correlated  LIWC with negative posts:")
    display(df_corr.sort_values("Pearson_r", ascending=False).head(3)[["Feature", "Pearson_r", "p_value"]])

    print("\n Top 3 negatively correlated LIWC with negative posts:")
    display(df_corr.sort_values("Pearson_r", ascending=True).head(3)[["Feature", "Pearson_r", "p_value"]])

def liwc_multivariate_analysis(df, properties_name):
    """
    Make multivariate analysis on the data

    :param df: pandas DataFrame of the Reddit data
    :param properties_name: all linguistic characteristics that we want to study
    """
    features = [f for f in properties_name if f != 'is_negative']
    X = df[features].fillna(0)  
    y = df['is_negative']

    #PCA - Dimensionality reduction to visualize data in 2D
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['is_negative'] = y.values

    # Visualize PCA results
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='is_negative',
                    palette={0:'green', 1:'red'}, alpha=0.7)
    plt.title('PCA: LIWC features')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.legend(title='Negative Post')
    plt.tight_layout()
    plt.show()

    #Clustering with kmeans to identify natural groupings
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters

    # Visualize clustering results
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters,
                    palette='Set2', alpha=0.7)
    plt.title('K-means Clustering on LIWC features')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

    #Random forest to identify most important features for predicting negativity
    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=features)
    importances = importances.sort_values(ascending=False)

    # Visualize feature importance
    plt.figure(figsize=(10,15))
    sns.barplot(x=importances.values, y=importances.index, palette='viridis')
    plt.title('Random Forest Feature Importance (predicting negative posts)')
    plt.xlabel('Importance')
    plt.ylabel('LIWC Feature')
    plt.tight_layout()
    plt.show()