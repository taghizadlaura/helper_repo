import plotly.io as pio
pio.renderers.default = "vscode"   # works best in VSCode
from calendar import month_name
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


current_dir = os.getcwd()


# Year plot of negative interactions with peak detection
def plot_yearly_evolution(df):

    # Filter negative interactions
    df_neg = df[df["LINK_SENTIMENT"] == -1].copy()
    df_neg["year"] = df_neg["TIMESTAMP"].dt.year
    # Exclude 2013 due to incomplete data
    df_neg = df_neg[df_neg["year"] != 2013]

    # Prepare data per year
    year_data = {}

    for year, df_year in df_neg.groupby("year"):

        # Daily aggregation
        daily_neg = (
            df_year.groupby(pd.Grouper(key="TIMESTAMP", freq="D"))
                .size()
                .rename("neg_count")
                .reset_index()
        )
        daily_neg["neg_count"] = daily_neg["neg_count"].fillna(0)

        # Peak detection
        peaks, properties = find_peaks(
            daily_neg["neg_count"],
            distance=3,
            prominence=5
        )
        peak_dates = daily_neg.loc[peaks, "TIMESTAMP"]

        # Peak details
        peak_info = {}
        for date in peak_dates:
            day_df = df_year[df_year["TIMESTAMP"].dt.date == date.date()]
            top_sources = day_df["SOURCE_SUBREDDIT"].value_counts().head(5).index.tolist()
            top_targets = day_df["TARGET_SUBREDDIT"].value_counts().head(5).index.tolist()

            peak_info[date] = {
                "Top Sources": ", ".join(top_sources),
                "Top Targets": ", ".join(top_targets)
            }

        # Hover text
        hover_text = []
        for i, row in daily_neg.iterrows():
            date = row["TIMESTAMP"]
            if date in peak_info:
                txt = (
                    f"<b>{date.date()}</b><br>"
                    f"Negative posts: {row['neg_count']}<br><br>"
                    f"<b>Top attacking subreddits:</b><br>{peak_info[date]['Top Sources']}<br><br>"
                    f"<b>Top targeted subreddits:</b><br>{peak_info[date]['Top Targets']}"
                )
            else:
                txt = f"{date.date()}<br>Negative posts: {row['neg_count']}"
            hover_text.append(txt)

        year_data[year] = {
            "daily_neg": daily_neg,
            "peaks": peaks,
            "hover": hover_text
        }

    # Build interactive plot

    fig = go.Figure()

    years = sorted(year_data.keys())

    # Create a trace for each year but initially hide them
    for i, year in enumerate(years):
        d = year_data[year]

        # Line trace
        fig.add_trace(go.Scatter(
            x=d["daily_neg"]["TIMESTAMP"],
            y=d["daily_neg"]["neg_count"],
            mode='lines',
            name=f"{year} Negative Interactions",
            hovertext=d["hover"],
            hoverinfo="text",
            visible=(i == 0)
        ))

        # Peak trace
        fig.add_trace(go.Scatter(
            x=d["daily_neg"].loc[d["peaks"], "TIMESTAMP"],
            y=d["daily_neg"].loc[d["peaks"], "neg_count"],
            mode="markers",
            marker=dict(size=10, color="red"),
            name=f"{year} Peaks",
            hovertext=[d["hover"][j] for j in d["peaks"]],
            hoverinfo="text",
            visible=(i == 0)
        ))


    # Slider definition
    steps = []
    for i, year in enumerate(years):
        step = dict(
            method="update",
            label=str(year),
            args=[
                {"visible": [False] * (2 * len(years))},  # hide all traces
                {"title": f"Negative Interactions Over Time — {year}"}
            ]
        )
        # Show only the 2 traces corresponding to the selected year
        step["args"][0]["visible"][2*i] = True
        step["args"][0]["visible"][2*i + 1] = True

        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Year: "},  
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Negative Interaction Count",
        height=600,
        title=f"Negative Interactions Over Time — {years[0]}"
    )

    fig.show()

    file_path = os.path.join(current_dir, 'images', 'yearly_evolution.html')

    fig.write_html(file_path)

# Day analysis of negativity for some LIWC features
def plot_daily_evolution(df, timestamp_col="TIMESTAMP"):
    """
    Interactive Plotly figure showing mean LIWC and negativity per time-of-day.
    Users can select Morning/Afternoon/Evening/Night and see changes in LIWC features.
    
    Parameters:
    - df: DataFrame containing timestamp column and LIWC/text features
    - timestamp_col: name of the timestamp column
    
    Returns:
    - fig: Plotly Figure with dropdown to select time period
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    

    # Define time-of-day bins
    def time_bin(hour):
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 22:
            return "Evening"
        else:
            return "Night"
    df["hour"] = df[timestamp_col].dt.hour
    df["time_bin"] = df["hour"].apply(time_bin)
    
    # Define target LIWC features and negativity
    features = ["LIWC_Posemo", "LIWC_Negemo", "LIWC_Anger", "LIWC_Anx", "LIWC_Sad", "LIWC_Swear"]
    # Negativity interaction count
    if "LINK_SENTIMENT" in df.columns:
        features.append("Negative interaction")
        df["Negative interaction"] = (df["LINK_SENTIMENT"] == -1).astype(int)
    
    # Color map for features
    color_map = {     
        "LIWC_Posemo": "#f3c35f",
        "LIWC_Negemo": "#e13c29",
        "LIWC_Anger": "#e13c29",
        "LIWC_Anx": "#b99be5",
        "LIWC_Sad": "#85b8ed",
        "LIWC_Swear": "#8b6a9e", 
        "Negative interaction": "#e13c29"
    }

    # Compute mean per time_bin
    grouped = df.groupby("time_bin")[features].mean()
    
    # Build figure with dropdown with color and emoji maps
    fig = go.Figure()
    for period in grouped.index:
        fig.add_trace(go.Bar(
            x=[feat.replace("LIWC_", "") for feat in features],
            y=grouped.loc[period].values,
            marker=dict(color=[color_map.get(feat, "#888") for feat in features]),
            name=period,
            hovertemplate="%{x}: %{y:.4f}<extra></extra>"
        ))
    fig.update_layout(
        title="Mean LIWC Scores by Time of the day",
        xaxis_title="LIWC Feature",
        yaxis_title="Mean Score",
        template="plotly_white",
        updatemenus=[{
            "buttons": [
                {
                    "method": "update",
                    "label": period,
                    "args": [{"visible": [p == period for p in grouped.index]},
                             {"title": f"Mean LIWC Scores - {period}"}]
                } for period in grouped.index
            ],
            "direction": "down",
            "showactive": True,
            "x": 0.0,
            "y": 1.15,
            "xanchor": "left",
            "yanchor": "top"
        }]
    )
    fig.show()
    file_path = os.path.join(current_dir, 'images', 'daily_evolution.html')

    fig.write_html(file_path)


#Monthly analysis
def monthly_liwc_variation(df, timestamp_col="TIMESTAMP"):
    """
    Create an animated Plotly figure showing monthly mean LIWC feature values
    and percent variation month-to-month across the whole dataset (all Reddit).
    
    Parameters:
    - df: DataFrame with timestamps and LIWC columns
    - timestamp_col: column name with datetime info (default "TIMESTAMP")
    
    Returns:
    - periodic_emotions_diff: DataFrame of month-to-month % changes
    - fig: Plotly Figure (animated)
    """
    
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    
    # Target LIWC features
    target_features = ["LIWC_Posemo", "LIWC_Negemo", "LIWC_Anger", "LIWC_Anx", "LIWC_Sad", "LIWC_Swear"]
    
    # Detect actual column names
    available = set(df.columns)
    feature_map = {}
    for feat in target_features:
        candidates = [
            feat,
            f"liwc_{feat}",
            f"LIWC_{feat}",
            f"normalized_{feat}",
            f"normalized_{feat}_without_neutral",
            f"normalized_liwc_{feat}",
            f"normalized_liwc_{feat}_without_neutral"
        ]
        found = next((c for c in candidates if c in available), None)
        if found:
            feature_map[feat] = found
        else:
            feature_map[feat] = None
    missing = [k for k,v in feature_map.items() if v is None]
    if missing:
        raise ValueError(f"Missing LIWC columns: {missing}")
    
    liwc_cols = [feature_map[f] for f in target_features]
    
    # Month columns
    df["month"] = df[timestamp_col].dt.month
    df["month_name"] = df["month"].apply(lambda m: month_name[m])
    
    # Monthly mean LIWC scores
    periodic_emotions = df.groupby("month")[liwc_cols].mean().reindex(range(1,13)).fillna(0)
    periodic_emotions.index = [month_name[m] for m in periodic_emotions.index]
    
    # Month-to-month percent changes
    periodic_emotions_diff = periodic_emotions.pct_change().fillna(0) * 100
    # January cyclic change
    dec = periodic_emotions.iloc[-1]
    jan = periodic_emotions.iloc[0]
    cyclic_change = (jan - dec) / dec.replace(0, np.nan) * 100
    cyclic_change = cyclic_change.fillna(0)
    periodic_emotions_diff.iloc[0] = cyclic_change
    
    # Display names and colors
    display_names = {liwc_cols[i]: target_features[i].replace("LIWC_", "") for i in range(len(target_features))}
    emotion_colors = {
        "LIWC_Posemo": "#f3c35f",
        "LIWC_Negemo": "#e13c29",
        "LIWC_Anger": "#e13c29",
        "LIWC_Anx": "#b99be5",
        "LIWC_Sad": "#85b8ed",
        "LIWC_Swear": "#8b6a9e"
    }
    emotion_emojis = {
        "LIWC_Posemo": "😊",
        "LIWC_Negemo": "😠",
        "LIWC_Anger": "😠",
        "LIWC_Anx": "😨",
        "LIWC_Sad": "😢",
        "LIWC_Swear": "🤬"
    }
    
    # Build figure with two panels
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Monthly variation (%) vs previous month", "Mean LIWC scores (monthly)"),
                        specs=[[{"type":"scatter"}, {"type":"bar"}]])
    
    months_order = list(periodic_emotions.index)
    frames = []
    
    for month in months_order:
        means = periodic_emotions.loc[month]
        variations = periodic_emotions_diff.loc[month]

        frame_traces = []

        # --- SCATTER TRACES (LEFT PANEL) ---
        for col in liwc_cols:
            val = variations[col]
            msize = max(6, min(60, abs(val)*1.2 + 6))
            
            trace = go.Scatter(
                x=[display_names[col]],
                y=[val],
                mode="markers+text",
                marker=dict(
                    size=msize,
                    color=emotion_colors.get(col, "#888888"),
                    line=dict(width=1, color='black'),
                    symbol="triangle-up" if val >= 0 else "triangle-down"
                ),
                text=[f"{val:+.2f}% {emotion_emojis.get(col,'')}"],
                textposition="top center" if val >= 0 else "bottom center"
            )

            # Assign trace to subplot (row 1, col 1)
            trace.update(xaxis="x", yaxis="y")

            frame_traces.append(trace)

        # --- BAR TRACE (RIGHT PANEL) ---
        bar_trace = go.Bar(
            x=[display_names[c] for c in liwc_cols],
            y=means.values,
            marker=dict(color=[emotion_colors.get(c, "#888888") for c in liwc_cols]),
        )

        # Assign trace to subplot (row 1, col 2)
        bar_trace.update(xaxis="x2", yaxis="y2")

        frame_traces.append(bar_trace)

        frame = go.Frame(
            data=frame_traces,
            name=month
        )

        frames.append(frame)



    # Assign frames so animation works
    fig.frames = frames


    # Initial traces
    initial = frames[0]
    for trace in initial.data:
        if isinstance(trace, go.Bar):
            fig.add_trace(trace, row=1, col=2)
        else:
            fig.add_trace(trace, row=1, col=1)
    
    steps = []
    for month in months_order:
        steps.append({
            "method": "animate",
            "label": month,
            "args": [
                [month],
                {"frame": {"duration": 800, "redraw": True},
                "mode": "immediate",
                "transition": {"duration": 500}}
            ]
        })

    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [
                        None,
                        {"frame": {"duration": 800, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 500}}
                    ]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {"frame": {"duration": 0, "redraw": False},
                        "mode": "immediate"}
                    ]
                }
            ]
        }],
        sliders=[{
            "active": 0,
            "steps": steps,
            "currentvalue": {"prefix": "Month: "}
        }],


        title="Monthly LIWC Mean Scores and Month-to-Month Variations",
        template="plotly_white",
        showlegend=False,
        width=1100,
        height=550,
        margin=dict(t=120,b=80,l=60,r=60)
    )


    # Axes
    fig.update_yaxes(title_text="Variation (%)", row=1, col=1)
    y_min = periodic_emotions.min().min()
    y_max = periodic_emotions.max().max()
    fig.update_yaxes(title_text="Mean LIWC score", range=[min(0, y_min*0.9), y_max*1.15], row=1, col=2)
    
    fig.update_layout(annotations=initial.layout.annotations)
    
    fig.show()

    file_path = os.path.join(current_dir, 'images', 'monthly_evolution.html')

    fig.write_html(file_path)
