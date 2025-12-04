import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import time

current_dir = os.getcwd()

# Define categories for LIWC features only (excluding Basic Text Metrics)
liwc_categories = {
    "🧠 LIWC Grammar": [
        "LIWC_Funct", "LIWC_Article", "LIWC_Verbs", "LIWC_AuxVb", "LIWC_Adverbs",
        "LIWC_Prep", "LIWC_Conj", "LIWC_Negate", "LIWC_Quant", "LIWC_Numbers"
    ],
    "👥 Pronouns": [
        "LIWC_Pronoun", "LIWC_Ppron", "LIWC_I", "LIWC_We", "LIWC_You",
        "LIWC_SheHe", "LIWC_They", "LIWC_Ipron"
    ],
    "⏱️ Verb Tenses": [
        "LIWC_Past", "LIWC_Present", "LIWC_Future"
    ],
    "💬 Social Language": [
        "LIWC_Social", "LIWC_Family", "LIWC_Friends", "LIWC_Humans"
    ],
    "💗 Emotional Language": [
        "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo", "LIWC_Anx", "LIWC_Anger", "LIWC_Sad"
    ],
    "🔍 Cognitive Processes": [
        "LIWC_CogMech", "LIWC_Insight", "LIWC_Cause", "LIWC_Discrep",
        "LIWC_Tentat", "LIWC_Certain", "LIWC_Inhib", "LIWC_Incl", "LIWC_Excl"
    ],
    "👁️ Perceptual References": [
        "LIWC_Percept", "LIWC_See", "LIWC_Hear", "LIWC_Feel"
    ],
    "🧬 Biological References": [
        "LIWC_Bio", "LIWC_Health", "LIWC_Sexual", "LIWC_Ingest"
    ],
    "🌐 Space & Time": [
        "LIWC_Relativ", "LIWC_Motion", "LIWC_Space", "LIWC_Time"
    ],
    "💼 Topical Categories": [
        "LIWC_Work", "LIWC_Achiev", "LIWC_Leisure", "LIWC_Home",
        "LIWC_Money", "LIWC_Relig", "LIWC_Death"
    ],
    "🗣️ Discourse Style": [
        "LIWC_Assent", "LIWC_Dissent", "LIWC_Nonflu", "LIWC_Filler"
    ],
    "🚨 Special Categories": [
        "LIWC_Swear", "LIWC_title"
    ]
}

liwc_descriptions = {
    "LIWC_Funct": "Function Words - Basic grammatical functions",
    "LIWC_Pronoun": "All Pronouns - Total pronoun usage",
    "LIWC_Ppron": "Personal Pronouns - I, you, she, he, etc.",
    "LIWC_I": "First Person Singular - I, me, my",
    "LIWC_We": "First Person Plural - we, us, our",
    "LIWC_You": "Second Person - you, your",
    "LIWC_SheHe": "Third Person Singular - she, he, her, him",
    "LIWC_They": "Third Person Plural - they, them, their",
    "LIWC_Ipron": "Impersonal Pronouns - it, its, those",
    "LIWC_Article": "Articles - a, an, the",
    "LIWC_Verbs": "Verbs - Action words",
    "LIWC_AuxVb": "Auxiliary Verbs - is, have, can, will",
    "LIWC_Past": "Past Tense - talked, went, was",
    "LIWC_Present": "Present Tense - talk, go, is",
    "LIWC_Future": "Future Tense - will, gonna",
    "LIWC_Adverbs": "Adverbs - quickly, very, really",
    "LIWC_Prep": "Prepositions - in, on, at, with",
    "LIWC_Conj": "Conjunctions - and, but, because",
    "LIWC_Negate": "Negations - no, not, never",
    "LIWC_Quant": "Quantifiers - many, few, much",
    "LIWC_Numbers": "Numbers - 1, 2, 100",
    "LIWC_Swear": "Swear Words - curse words",
    "LIWC_Social": "Social Words - friend, talk, family",
    "LIWC_Family": "Family - mother, father, sister",
    "LIWC_Friends": "Friends - buddy, pal, friend",
    "LIWC_Humans": "Human References - people, person",
    "LIWC_Affect": "Affective Processes - emotional words",
    "LIWC_Posemo": "Positive Emotions - happy, good, love",
    "LIWC_Negemo": "Negative Emotions - sad, bad, hate",
    "LIWC_Anx": "Anxiety - nervous, worried, tense",
    "LIWC_Anger": "Anger - angry, hate, kill",
    "LIWC_Sad": "Sadness - sad, crying, grief",
    "LIWC_CogMech": "Cognitive Processes - think, know, consider",
    "LIWC_Insight": "Insight - understand, realize",
    "LIWC_Cause": "Causation - because, effect, hence",
    "LIWC_Discrep": "Discrepancy - should, would, could",
    "LIWC_Tentat": "Tentative - maybe, perhaps, guess",
    "LIWC_Certain": "Certainty - always, never",
    "LIWC_Inhib": "Inhibition - block, constrain",
    "LIWC_Incl": "Inclusive - we, with, together",
    "LIWC_Excl": "Exclusive - but, except, without",
    "LIWC_Percept": "Perceptual Processes - see, hear, feel",
    "LIWC_See": "Seeing - view, saw, look",
    "LIWC_Hear": "Hearing - hear, listen, sound",
    "LIWC_Feel": "Feeling - feel, touch, hold",
    "LIWC_Bio": "Biological Processes - eat, blood, pain",
    "LIWC_title": "Title Words - Mr., Mrs., Dr.",
    "LIWC_Health": "Health - medical, clinic, pill",
    "LIWC_Sexual": "Sexual - love, sex, kiss",
    "LIWC_Ingest": "Ingestion - eat, drink, swallow",
    "LIWC_Relativ": "Relativity - area, bend, stop",
    "LIWC_Motion": "Motion - go, come, carry",
    "LIWC_Space": "Space - up, down, in",
    "LIWC_Time": "Time - hour, day, year",
    "LIWC_Work": "Work - job, boss, career",
    "LIWC_Achiev": "Achievement - win, success, better",
    "LIWC_Leisure": "Leisure - house, TV, music",
    "LIWC_Home": "Home - house, room, door",
    "LIWC_Money": "Money - cash, buy, sell",
    "LIWC_Relig": "Religion - church, God, prayer",
    "LIWC_Death": "Death - dead, kill, bury",
    "LIWC_Assent": "Assent - yes, okay, agree",
    "LIWC_Dissent": "Dissent - no, not, never",
    "LIWC_Nonflu": "Non-fluencies - er, hm, uh",
    "LIWC_Filler": "Fillers - I mean, you know"
}

def create_liwc_sunburst(df, LIWC_list):
    """Create a sunburst chart for LIWC features only"""
    
    # Get available LIWC features (excluding basic text metrics)
    liwc_features = [f for f in LIWC_list if f in df.columns and np.issubdtype(df[f].dtype, np.number)]
    all_liwc_available = liwc_features 
    
    # Calculate percentages for LIWC features only
    feature_means = {f: df[f].mean() for f in all_liwc_available}
    total_liwc = sum(feature_means.values())
    
        
    feature_percentages = {k: (v/total_liwc)*100 for k, v in feature_means.items()}
    
    # Prepare data for sunburst
    labels = ['LIWC Features']
    parents = ['']
    values = [100]
    custom_data = [['All LIWC linguistic features']]
    
    # Color palette for LIWC categories
    liwc_category_colors = {
        "🧠 LIWC Grammar": '#2ca02c',
        "👥 Pronouns": '#d62728',
        "⏱️ Verb Tenses": '#9467bd',
        "💬 Social Language": '#8c564b',
        "💗 Emotional Language": '#e377c2',
        "🔍 Cognitive Processes": '#7f7f7f',
        "👁️ Perceptual References": '#bcbd22',
        "🧬 Biological References": '#17becf',
        "🌐 Space & Time": '#ff9896',
        "💼 Topical Categories": '#aec7e8',
        "🗣️ Discourse Style": '#ffbb78',
        "🚨 Special Categories": '#98df8a'
    }
    
    colors = ['#2ca02c']
    
    # Add LIWC categories
    for category, features in liwc_categories.items():
        # Calculate category total
        category_features = [f for f in features if f in feature_percentages]
        if category_features:
            category_value = sum(feature_percentages[f] for f in category_features)
            
            if category_value > 0.001:
                labels.append(category)
                parents.append('LIWC Features')
                values.append(category_value)
                custom_data.append([f'{category} features'])
                colors.append(liwc_category_colors.get(category, 'lightgreen'))
                
                # Add individual features within category
                for feature in category_features:
                    if feature in feature_percentages and feature_percentages[feature] > 0.001:
                        labels.append(feature)
                        parents.append(category)
                        values.append(feature_percentages[feature])
                        description = liwc_descriptions.get(feature, "No description available")
                        custom_data.append([description])
                        base_color = liwc_category_colors.get(category, 'lightgreen')
                        colors.append(base_color)
    
    # Create sunburst chart
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        customdata=custom_data,
        hovertemplate='<b>%{label}</b><br>Percentage: %{value:.3f}%<br>Description: %{customdata[0]}<extra></extra>',
        marker=dict(colors=colors),
        branchvalues="total",
        maxdepth=3
    ))
    
    fig.update_layout(
        title={
            'text': '🧠 LIWC Linguistic Features<br><sub>Click to explore categories',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=800,
        margin=dict(t=100, l=0, r=0, b=0),
        font=dict(size=12)
    )
    
    fig.add_annotation(
        text="💡 <b>LIWC Features Only</b><br>Linguistic and psychological analysis",
        xref="paper", yref="paper",
        x=0.02, y=0.02,
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Save HTML file
    file_path = os.path.join(current_dir, 'images', 'sunburst_liwc.html')
    fig.write_html(
        file_path)
    
    # Show the figure
    fig.show()
    

