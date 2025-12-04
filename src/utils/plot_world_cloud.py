import pandas as pd
import numpy as np
#!pip install wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def dictionary_world_cloud(df, LIWC_list, link):
    # Split positive and negative posts
    df_bis = df[df['LINK_SENTIMENT'] == link]


    # Sum features across posts to get total “weight” per feature
    feature_sums = df_bis[LIWC_list].sum()

    # Keep top 10 features
    top10 = feature_sums.sort_values(ascending=False).head(10)

    # Prepare dictionary for WordCloud
    dict = top10.to_dict()


    return dict

def plot_feature_wordcloud(df, LIWC_list, title,name, link_sentiment, color):

    feature_dict = dictionary_world_cloud(df, LIWC_list, link_sentiment)

    wc = WordCloud(width=600, height=400, background_color='white', colormap=color)
    wc.generate_from_frequencies(feature_dict)
    
    plt.figure(figsize=(8, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.savefig("images/"+name)
    plt.show()


