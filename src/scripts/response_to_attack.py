import pandas as pd
from IPython.display import display
import numpy as np
from scipy.stats import ttest_rel
from scipy.stats import t

def attack_answer_analysis(df):
    print("\n Anger and negativity level difference from the exchange of the victim subreddit before and the answer to the attack")
    df["is_negative"] = (df['LINK_SENTIMENT'] == -1).astype(int)

    attacks = df[df['LINK_SENTIMENT'] == -1]
    valid_attacks = []

    for idx, attack in attacks.iterrows():
        
        #Get the 3 most recent exchange that the target subreddit has send before
        target_before = df[
            (df['SOURCE_SUBREDDIT'] == attack['TARGET_SUBREDDIT']) &
                (df['TIMESTAMP'] < attack['TIMESTAMP'])
                    ].sort_values('TIMESTAMP', ascending=False).head(3)
        
        #Get the most recent exchange that the target subreddits has re send after 
        target_after = df[(df['SOURCE_SUBREDDIT'] == attack['TARGET_SUBREDDIT']) & (df["TARGET_SUBREDDIT"]== attack["SOURCE_SUBREDDIT"]) & 
                         (df['TIMESTAMP'] > attack['TIMESTAMP'])].sort_values('TIMESTAMP', ascending = False).head(1)
                                                                                                                
        if len(target_before) > 0 and len(target_after) > 0:
            valid_attacks.append((idx, attack))
    
        #too long if more :()
        if len(valid_attacks)>20:
            break

    results = []
    
    #Only calculate the mean if valid exchange before and after 
    for idx, attack in valid_attacks:

        #Get the 3 most recent exchange that the target subreddit has send before
        before_attack = df[(df['SOURCE_SUBREDDIT'] == attack['TARGET_SUBREDDIT']) &
                (df['TIMESTAMP'] < attack['TIMESTAMP'])
                    ].sort_values('TIMESTAMP', ascending=False).head(3)
        
        #Get the most recent exchange that the target subreddits has re send after 
        after_attack = df[(df['SOURCE_SUBREDDIT'] == attack['TARGET_SUBREDDIT']) & (df["TARGET_SUBREDDIT"]== attack["SOURCE_SUBREDDIT"]) & 
                         (df['TIMESTAMP'] > attack['TIMESTAMP'])].sort_values('TIMESTAMP', ascending = False).head(1)

        LIWC_before = before_attack["LIWC_Anger"].mean()
        negativity_before= before_attack["is_negative"].mean()
        LIWC_after =  after_attack["LIWC_Anger"].iloc[0]
        negativity_after= after_attack["is_negative"].iloc[0]
                                                            
        results.append({
            'attack_id': idx,
            'anger_before': LIWC_before,
            'negativity_before': negativity_before,
            'anger_after': LIWC_after,
            'negativity_after' : negativity_after
           
        })
        


    results_df = pd.DataFrame(results)
    print(f"\n ===== Targetted subreddits have a mean LIWC of anger ===== ")
    print(f"\n Before the attack (last 3 exchanges with the source) of {results_df['anger_before'].mean()} ")
    print(f"\n After the attack (next exchange with the source) of {results_df['anger_after'].mean()} ")
    print(f"\n===== Targetted subreddits have a mean negativity ===== ")
    print(f"\n Before the attack (last 3 exchanges with the source) of {results_df['negativity_before'].mean()} ")
    print(f"\n After the attack (next exchange with the source) of {results_df['negativity_after'].mean()} ")
  
    return results_df


def attack_exchange_analysis(df):
    print("\n Anger and negativity level before and after a conflict in the exchange from the victim to the attacker")
    df["is_negative"] = (df['LINK_SENTIMENT'] == -1).astype(int)

    attacks = df[df['LINK_SENTIMENT'] == -1]
    valid_attacks = []

    for idx, attack in attacks.iterrows():
        
        #Get the 3 most recent exchange that the target subreddit has send before to the attacker
        target_before = df[
            (df['SOURCE_SUBREDDIT'] == attack['TARGET_SUBREDDIT']) & (df["TARGET_SUBREDDIT"]== attack["SOURCE_SUBREDDIT"]) & 
                (df['TIMESTAMP'] < attack['TIMESTAMP'])
                    ].sort_values('TIMESTAMP', ascending=False).head(3)
        
        #Get the most recent exchange that the target subreddits has re send after 
        target_after = df[(df['SOURCE_SUBREDDIT'] == attack['TARGET_SUBREDDIT']) & (df["TARGET_SUBREDDIT"]== attack["SOURCE_SUBREDDIT"]) & 
                         (df['TIMESTAMP'] > attack['TIMESTAMP'])].sort_values('TIMESTAMP', ascending = False).head(1)
                                                                                                                
        if len(target_before) > 0 and len(target_after) > 0:
            valid_attacks.append((idx, attack))
    
        #too long if more :()
        if len(valid_attacks)>20:
            break

    results = []
    
    #Only calculate the mean if valid exchange before and after 
    for idx, attack in valid_attacks:

        #Get the 3 most recent exchange that the target subreddit has send before to the attacker
        before_attack = df[(df['SOURCE_SUBREDDIT'] == attack['TARGET_SUBREDDIT']) & (df["TARGET_SUBREDDIT"]== attack["SOURCE_SUBREDDIT"]) & 
                (df['TIMESTAMP'] < attack['TIMESTAMP'])
                    ].sort_values('TIMESTAMP', ascending=False).head(3)
        
        #Get the most recent exchange that the target subreddits has re send after 
        after_attack = df[(df['SOURCE_SUBREDDIT'] == attack['TARGET_SUBREDDIT']) & (df["TARGET_SUBREDDIT"]== attack["SOURCE_SUBREDDIT"]) & 
                         (df['TIMESTAMP'] > attack['TIMESTAMP'])].sort_values('TIMESTAMP', ascending = False).head(1)

        LIWC_before = before_attack["LIWC_Anger"].mean()
        negativity_before= before_attack["is_negative"].mean()
        LIWC_after =  after_attack["LIWC_Anger"].iloc[0]
        negativity_after= after_attack["is_negative"].iloc[0]
                                                            
        results.append({
            'attack_id': idx,
            'anger_before': LIWC_before,
            'negativity_before': negativity_before,
            'anger_after': LIWC_after,
            'negativity_after' : negativity_after
           
        })
        


    results_df = pd.DataFrame(results)
    print(f"\n ===== Targetted subreddits have a mean LIWC of anger ===== ")
    print(f"\n Before the attack (last 3 exchanges with the source) of {results_df['anger_before'].mean()} ")
    print(f"\n After the attack (next exchange with the source) of {results_df['anger_after'].mean()} ")
    print(f"\n===== Targetted subreddits have a mean negativity ===== ")
    print(f"\n Before the attack (last 3 exchanges with the source) of {results_df['negativity_before'].mean()} ")
    print(f"\n After the attack (next exchange with the source) of {results_df['negativity_after'].mean()} ")
  
    return results_df

def test_significativity(results_df):
    print("\n ==== Statistical test for anger level ====")
    anger_before_all = results_df['anger_before']
    anger_after_all = results_df['anger_after']
    
    #Paired t-test to test if significance between before and after
    t_stat_anger, p_value_anger = ttest_rel(anger_before_all, anger_after_all)
    
    print(f"\n The P-value of the paired T-test for anger level is: {p_value_anger}")

    #Test the magnitude of the change with Cohen test
    def cohens_d_paired(x, y):
        diff = x - y
        return np.mean(diff) / np.std(diff, ddof=1)
    
    anger_effect_size = cohens_d_paired(anger_after_all, anger_before_all)

    print(f"\n The effect size of the conflict for anger level is : {anger_effect_size:.4f}")

    print("\n ==== Statistical test for negativity level ====")
    negativity_before_all = results_df['negativity_before']
    negativity_after_all = results_df['negativity_after']
    
    #Paired t-test to test if significance between before and after
    t_stat_negativity, p_value_negativity = ttest_rel(negativity_before_all, negativity_after_all)
    
    print(f"\n The P-value of the paired T-test for negativity level is: {p_value_negativity }")

    #Test the magnitude of the change with Cohen test
    def cohens_d_paired(x, y):
        diff = x - y
        return np.mean(diff) / np.std(diff, ddof=1)
    
    negativity_effect_size = cohens_d_paired(negativity_after_all, negativity_before_all)

    print(f"\n The effect size of the conflict for negativity level is : {negativity_effect_size:.4f}")