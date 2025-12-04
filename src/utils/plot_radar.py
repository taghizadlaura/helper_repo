
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict





def plot_radar(cluster_name, data, feature_names):

    values = data.values.flatten().tolist()
    values += values[:1]   # close the loop

    num_vars = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=11)
    ax.set_yticklabels([])

    ax.set_title(f"Cluster {cluster_name}: LIWC & Text Profile", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()




import pandas as pd
import numpy as np
import random
from collections import defaultdict
import ipywidgets as widgets
import plotly.graph_objects as go
from ipywidgets import interact, widgets, VBox, HBox, Button, HTML
from ipywidgets import Button, VBox, HTML
import plotly.io as pio

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

def quiz(cluster_norm, cluster_to_subreddits, properties_name): 
        
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
        fig = radar_plot(correct, properties_name, cluster_norm)
        pio.show(fig, notebook=True)

        # Create answer buttons
        buttons = []
        def make_handler(choice):
            def handler(_):
                if choice == correct:
                    out.value = (
                        f"<span style='color:green;font-weight:bold'>Correct!</span>"
                        f"<br>Cluster {choice} includes:<br>{cluster_to_subreddits[choice]}"
                    )
                else:
                    out.value = (
                        f"<span style='color:red;font-weight:bold'>Wrong.</span>"
                        f"<br>The correct answer was cluster {correct}"
                        f"<br>({cluster_to_subreddits[correct]})"
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
