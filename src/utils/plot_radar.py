
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import ipywidgets as widgets
import plotly.graph_objects as go
from ipywidgets import interact, widgets, VBox, HBox, Button, HTML
from ipywidgets import Button, VBox, HTML
import plotly.io as pio
import json
import plotly.graph_objects as go
import os
import random

def radar_plot(cluster_id, properties_name, cluster_norm):
    values = cluster_norm.loc[cluster_id].values.tolist()
    values += values[:1]  # loop closure

    angles = list(range(len(properties_name)))
    feature_names = properties_name + [properties_name[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=feature_names,
        fill='toself',
        name=f"Cluster {cluster_id}"
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False)),
        showlegend=False,
        title=f"Guess which cluster this is!"
    )
    return fig

Button(description="Test")

def quiz(cluster_norm, cluster_to_subreddits): 
    LIWC_list = [
        "LIWC_Negate", "LIWC_Swear", "LIWC_Social", "LIWC_Family",
        "LIWC_Friends", "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo",
        "LIWC_Anx", "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech", "LIWC_Sexual",
        "LIWC_Relativ", "LIWC_Achiev", "LIWC_Leisure", "LIWC_Money",
        "LIWC_Relig", "LIWC_Death"
    ]
    # Implement quiz logic

    clusters = list(cluster_norm.index)
    current_correct_answer = [None]  # mutable container

    def generate_question():
        correct = random.choice(clusters)

        # 3 random incorrect answers
        wrong = random.sample([c for c in clusters if c != correct], 3)

        # Shuffle options
        options = wrong + [correct]
        random.shuffle(options)

        current_correct_answer[0] = correct

        return correct, options


    def ask_new_question(_=None):
        # Generate question
        correct, options = generate_question()

        # Output areas
        out = HTML(value="<b>Which cluster does this radar plot represent?</b>")

        # Show radar plot of the correct cluster
        fig = radar_plot(correct, LIWC_list, cluster_norm)
        pio.show(fig, notebook=True)

        # Create answer buttons
        buttons = []
        def make_handler(choice):
            def handler(_):
                if choice == correct:
                    out.value = (
                        f"<span style='color:green;font-weight:bold'>Correct!</span>"
                        f"<br>Cluster {choice} includes:<br>{cluster_to_subreddits[correct][:10]}"
                    )
                else:
                    out.value = (
                        f"<span style='color:red;font-weight:bold'>Wrong.</span>"
                        f"<br>The correct answer was cluster {correct}"
                        f"<br>({cluster_to_subreddits[correct][:10]})"
                    )
            return handler

        for opt in options:
            btn = Button(description=str(opt), layout=widgets.Layout(width="120px"))
            btn.on_click(make_handler(opt))
            buttons.append(btn)

        # Next question button
        next_btn = Button(description="Next Question", button_style="info")
        next_btn.on_click(ask_new_question)

        display(VBox([
            out,
            HBox(buttons),
            next_btn
        ]))

        # Launch the quiz
    ask_new_question()
 

def save_static_quiz(cluster_norm, cluster_to_subreddits,outdir="quiz_static"):
    LIWC_list = [
        "LIWC_Negate", "LIWC_Swear", "LIWC_Social", "LIWC_Family",
        "LIWC_Friends", "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo",
        "LIWC_Anx", "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech", "LIWC_Sexual",
        "LIWC_Relativ", "LIWC_Achiev", "LIWC_Leisure", "LIWC_Money",
        "LIWC_Relig", "LIWC_Death"
    ]

    os.makedirs(outdir, exist_ok=True)

    clusters = list(cluster_norm.index)

    def make_fig(cid):
        values = cluster_norm.loc[cid].tolist()
        values = values + [values[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta= LIWC_list + [LIWC_list[0]],
            fill="toself"
        ))
        fig.update_layout(
            title=f"Cluster {cid}",
            polar=dict(radialaxis=dict(visible=False)),
            showlegend=False
        )
        return fig

    # -------------------------------
    # Generate 50 static questions
    # -------------------------------
    NUM_Q = 50
    q_files = []

    for qi in range(NUM_Q):
        correct = random.choice(clusters)
        wrong = random.sample([c for c in clusters if c != correct], 3)
        opts = wrong + [correct]
        random.shuffle(opts)

        # Make plot JSON
        fig = make_fig(correct)
        fig_json = fig.to_json()

        qfile = f"q_{qi:04d}.html"
        q_path = os.path.join(outdir, qfile)
        q_files.append(qfile)

        html = f"""
<html>
<head>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <meta charset="utf-8"/>
</head>

<body style="font-family:Arial; margin:40px">

<h2>Which cluster does this radar plot represent?</h2>

<div id="plot"></div>

<script>
var fig = {fig_json};
Plotly.newPlot('plot', fig.data, fig.layout);
</script>

<h3>Choose:</h3>
"""

        for opt in opts:
            html += f"""
<div style="margin:5px">
  <a href="answer_{qi:04d}_{opt}.html">
    <button style="padding:10px 20px; font-size:16px">{opt}</button>
  </a>
</div>
"""

        html += "</body></html>"

        with open(q_path, "w", encoding="utf-8") as f:
            f.write(html)

        # -------------------------------
        # Create answer pages
        # -------------------------------
        for opt in opts:
            is_correct = (opt == correct)
            subs = "<br>".join(cluster_to_subreddits[correct][:10])

            ans_file = f"answer_{qi:04d}_{opt}.html"
            ans_path = os.path.join(outdir, ans_file)

            ans_html = f"""
<html><body style="font-family:Arial; margin:40px">
<h2>{'Correct!' if is_correct else 'Wrong.'}</h2>

<p>
Correct cluster was <b>{correct}</b><br>
Top subreddits:<br>
{subs}
</p>

<a href="{qfile}">
  <button style="padding:10px 20px">Try again</button>
</a>

<a href="index.html">
  <button style="padding:10px 20px; margin-left:20px">Back to Home</button>
</a>

</body></html>
"""

            with open(ans_path, "w", encoding="utf-8") as f:
                f.write(ans_html)

    # -------------------------------
    # Create index.html
    # -------------------------------
    index_html = "<html><body><h1>Cluster Quiz</h1><ul>"
    for qf in q_files:
        index_html += f'<li><a href="{qf}">{qf}</a></li>'
    index_html += "</ul></body></html>"

    with open(os.path.join(outdir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)

    print(f"Static quiz saved to folder: {outdir}")
