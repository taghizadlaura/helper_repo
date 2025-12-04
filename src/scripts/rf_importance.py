import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def random_forest(df, text_features, liwc_features):
    X_text = df[text_features].fillna(0) if text_features else pd.DataFrame()
    X_liwc = df[liwc_features].fillna(0) if liwc_features else pd.DataFrame()
    y = df['is_negative']
    

    rf_text = RandomForestClassifier(n_estimators=500, random_state=42)
    rf_text.fit(X_text, y)
    importances_text = pd.Series(rf_text.feature_importances_, index=text_features)
    importances_text = importances_text.sort_values(ascending=False)
        
    fig1 = px.bar(
            x=importances_text.values,
            y=importances_text.index,
            orientation='h',
            title=f'<b>Text Features Importance</b><br><sub>{len(text_features)} features</sub>',
            labels={'x': 'Importance', 'y': 'Feature'},
            color=importances_text.values,
            color_continuous_scale='blues'
    )
    fig1.update_layout(height=600, showlegend=False, yaxis={'categoryorder': 'total ascending'})
    fig1.show()
    
    fig1.write_html(
    'rf-text--importance.html',
    include_plotlyjs=True,
    config={
        'responsive': True,
        'displayModeBar': True
    },
    auto_open=False  # Don't automatically open in browser
    )

    rf_liwc = RandomForestClassifier(n_estimators=500, random_state=42)
    rf_liwc.fit(X_liwc, y)
    importances_liwc = pd.Series(rf_liwc.feature_importances_, index=liwc_features)
    importances_liwc = importances_liwc.sort_values(ascending=False)
        
    fig2 = px.bar(
            x=importances_liwc.values,
            y=importances_liwc.index,
            orientation='h',
            title=f'<b>LIWC Features Importance</b><br><sub>{len(liwc_features)} features</sub>',
            labels={'x': 'Importance', 'y': 'Feature'},
            color=importances_liwc.values,
            color_continuous_scale='reds'
    )
    fig2.update_layout(height=600, showlegend=False, yaxis={'categoryorder': 'total ascending'})
    fig2.show()

    fig2.write_html(
    'rf-importance.html',
    include_plotlyjs=True,
    config={
        'responsive': True,
        'displayModeBar': True
    },
    auto_open=False  # Don't automatically open in browser
    )
    
    return {
        'text_features': importances_text if len(text_features) > 0 else pd.Series(),
        'liwc_features': importances_liwc if len(liwc_features) > 0 else pd.Series()
    }
