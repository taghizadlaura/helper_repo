
# Linguistic and Emotional Dynamics of Reddit Communities: From Inter-Community Conflicts to Global Political Event


## Abstract

In the digital realm, words spread faster than intentions. A single post can ignite tension between distant communities, leaving traces that reshape their emotional landscapes. This project tells the story of how online hostility unfolds, lingers, and transforms the language of those it touches. Focusing on Reddit communities from 2014 to 2017, the project explores how conflicts between subreddits reveal deeper patterns of collective emotion, examining how anger spreads and how communities recover. It also considers how repeated clashes may harden or exhaust communities over time. By tracing these emotional aftershocks, the project seeks to understand not just moments of division, but the fragile processes of linguistic healing that follow. Beyond individual disputes, it asks how local conflicts mirror broader societal shifts, from everyday tensions to global political events. The aim is to uncover what our digital language reveals about resilience, contagion, and the shared emotional life of online communities.

## Quickstart

```bash
# clone project
git clone <project link>
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>


# install requirements
pip install -r pip_requirements.txt
```


### How to use the library
Clone the project using the above code. Download the data files from [this website](https://snap.stanford.edu/data/soc-RedditHyperlinks.html) and [that one](https://snap.stanford.edu/data/web-RedditEmbeddings.html). Create the `data` folder and add the data files to it. The necessary libraries are in the `pip_requirements.txt` file.


## Project Structure

The directory structure of new project looks like this:

```
├── data
│   ├── soc-redditHyperlinks-body.tsv
│   ├── soc-redditHyperlinks-title.tsv
│   ├── web-redditEmbeddings-subreddits.csv
│   └── web-redditEmbeddings-users.csv
├── src
│   ├── data
│   │   └── dataloader.py
│   ├── models
│   │   ├── hdbscan_emb.py
│   │   ├── kmeans_emb.py
│   │   └── louvain.py
│   ├── scripts
│   │   ├── conflict_definition.py
│   │   ├── conflict_id.py
│   │   ├── LIWC_statistical_analaysis.py
│   │   ├── response_to_attack.py
│   │   ├── subreddits_activity.py
│   │   ├── temporal_analysis.py
│   │   └── t_test_deltaLIWC.py
│   └── utils
│       └── plot_utils.py
├── tests
│   └── test_rf_deltaLIWC.py
├── .gitignore
├── README.md
├── open_file.py
├── pip_requirements.txt
└── results.ipynb
```


## Research questions
- Words as weapons: how do inter-subreddit conflicts affect victims’ linguistic recovery?
  - How does the conflict LIWC characteristics impact the afterward interaction tones of the targeted subreddit ?
  - How long does it take for the linguistic tone of a “victim” subreddit to return to its baseline after being targeted in a conflict?
- Snowball effect:  How do repeated inter-subreddit conflicts shape the linguistic and emotional tone (LIWC features) of both attackers and targets over time, and do these   patterns suggest a cumulative negativity ?
  - Do the linguistic and emotional characteristics (as measured by LIWC) of a targeted subreddit’s responses differ when replying to an attacking subreddit compared to their           usual interactions?
  - Does the intensity or frequency of conflicts lead to a cumulative increase in aggressive or negative linguistic traits ?
- How do the linguistic profiles of online community clusters, as captured through LIWC features, relate to and predict the propagation of conflict between communities?
  - What are the distinctive LIWC profiles of different community clusters during periods of normal interaction?
  - How do these linguistic characteristics predict the conflict propagation between clusters between negative interactions?
- How do posts related to U.S.A. politics evolve before vs after the elections of 2016?
  - To what extent do these changes reflect collective reactions to major political events, such as the 2016 U.S.A. presidential election?
  - How does linguistic similarity between political subreddits evolve - decrease or increase - over time, particularly around major political events?


## Methods
We want to find answers to the above research questions. To define a conflict we will look at source-to-target and inter-communities interactions. We then find the impact of these conflicts on the other subreddits posts / communities. From these conflicts, we can link them the major conflicts to events that happened in the U.S.A. 

### Data overview 

- **Descriptive Analysis**: We will profile the data by comparing the volume and nature of negative vs. positive hyperlinks.
- **Statistical Testing**: We will perform a T-test to determine if there is a statistically significant difference in the LIWC features (e.g., anger, anxiety) of posts containing negative hyperlinks versus those containing positive ones.

### Analysis of Conflict Impact
- **Conflict Definition**: A "conflict" is formally operationalized as a documented negative hyperlink from a source subreddit to a target subreddit.
- **Linguistic Impact on Victims**: For each conflict event, we will analyze the LIWC features of the victim community's posts before and after the attack. A paired T-test will be used to identify significant shifts in emotional and cognitive language.
- **Predicting Impact Severity**: Using a Random Forest model, we will determine which specific LIWC features of the attacker's post are the most important predictors of the magnitude of linguistic change in the victim community. This identifies the linguistic "weapons" that cause the most damage.
- **Snowball effect**: Using causal and statistical analysis of LIWC metrics, we will asses wether repeated attacks lead to a cumulative increase in negativity.

### Macro-Level Cluster Analysis
- **Cluster Definition**: Communities will be grouped into clusters using an unsupervised method (e.g., based on shared membership, topic modeling, or network connectivity). These clusters represent macro-communities or "alliances."
- **Cluster Profiling**:  Each cluster will be assigned a baseline LIWC profile from its internal, positive interactions.
- **Cross-Cluster Conflict**: We will analyze inter-cluster conflicts to see if certain cluster pairings (e.g., Cluster A attacking Cluster B) consistently produce stronger negative linguistic outcomes.

### U.S.A. Events Analysis
- **Relate to real events**: Analyse U.S.A. major events from 2014 to 2017.
- **Association with reddit dynamics**: We will associate peaks in negativity to real-world events.

### Website (M3)
(*To be developed when we get the criteria)*


## Proposed timeline
| Dates | Charlotte W. | Miyuka L. | Mathilde B.                                            | Laura T.                                                | France L.                                               | Milestones                                                                                                                                    |
|-------|----|----|--------------------------------------------------------|---------------------------------------------------------|---------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Thursday Nov 6 - Wednesday Nov 12  | Cluster Profiling, Cross-Cluster Conflict | Linguistic Impact on Victims, Predicting Impact Severity | Linguistic Impact on Victims, Predicting Impact Severity | Cluster Profiling, Cross-Cluster Conflict               | Cluster Profiling, Cross-Cluster Conflict                      | -                                                                                                                                             |
| Thursday Nov 13 - Wednesday Nov 19 | Cross-Cluster Conflict | Linguistic Impact on Victims, Predicting Impact Severity | Linguistic Impact on Victims, Predicting Impact Severity | Cross-Cluster Conflict                                  | Cluster Profiling, Cross-Cluster Conflict                      | **Sunday Nov 16:** Finalise the implementations of the different conflict definitions, in order to start working on conflict-related effects. |
| Thursday Nov 20 - Wednesday Nov 26 | Snowball effect | Snowball effect | Relate to real events, Association with reddit dynamics | Relate to real events, Association with reddit dynamics | Relate to real events, Association with reddit dynamics | -                                                                                                                                             |
| Thursday Nov 27 - Wednesday Dec 3  | Snowball effect | Snowball effect | Association with reddit dynamics                       | Association with reddit dynamics, Website           | Association with reddit dynamics, Website                    | **Sunday Nov 30:** Finalise all core data analysis and results. Begin development of the presentation website.                                |
| Thursday Dec 4 - Wednesday Dec 10  | Website | Website | Website                                                       | Website                                                 | Website                                                 | -                                                                                                                                             |
| Thursday Nov 11 - Wednesday Dec 17 | Finalise project | Finalise project | Finalise project                                       | Finalise project                                        | Finalise project                                        | **Sunday Dec 14:** Website finalized and deployed<br>**Wednesday Dec 17:** Milestone 3                                                        |


## Workload distribution
- Charlotte WANG: Cluster definition, development, and macro-level analysis of inter-cluster dynamics.
- Miyuka LAURENSON: Conflict definition, primary analysis of LIWC changes post-conflict, and Random Forest modeling to predict impact severity.
- Mathilde BOUSSEMART: GitHub repository management, project overview analysis, and specific analysis of the linguistic patterns in the victim community's direct responses to attackers.
- Laura TAGHIZAD: GitHub repository management, descriptive analysis, primary analysis of LIWC changes, cluster definition.
- France LU: GitHub repository management, descriptive analysis, project overview analysis.
