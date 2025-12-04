import plotly.graph_objects as go
import numpy as np
import pandas as pd

def plot_results(records):
    df_granger = pd.DataFrame(records)
    df_granger['minus_log10_p'] = -np.log10(df_granger['p_value']+1e-10)
    fig = go.Figure()

    for freq in ['H','D']:
        df_freq = df_granger[df_granger['frequency']==freq]
        pivot = df_freq.pivot_table(index='subreddit', columns='lag', values='minus_log10_p', aggfunc='max')
        
        fig.add_trace(go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='Viridis',
            zmin=0,
            visible=(freq=='D'),  # show only daily by default
            colorbar=dict(title='-log10(p-value)')
        ))

    # Dropdown buttons
    fig.update_layout(
        updatemenus=[    
            dict(
                buttons=[
                    dict(label='Hourly',
                        method='update',
                        args=[{'visible':[True, False]}]),
                    dict(label='Daily',
                        method='update',
                        args=[{'visible':[False, True]}]),
                ],
                direction="down"
            )
        ],
        title="Granger causality (Attacks → Negative behavior)",
        xaxis_title="Lag",
        yaxis_title="Subreddit"
    )

    fig.show()
    fig.write_html(
    'images/granger_analysis.html',
            include_plotlyjs=True,
            config={
                'responsive': True,
                'displayModeBar': True
            },
            auto_open=False)

