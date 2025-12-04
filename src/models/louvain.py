import networkx as nx
import community.community_louvain as community_louvain

def build_network(df, directed=True):
    """
    Create a NetworkX graph from a pandas DataFrame with columns 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', and 'LINK_SENTIMENT'.
    """
    G = nx.from_pandas_edgelist(
        df,
        'SOURCE_SUBREDDIT',
        'TARGET_SUBREDDIT',
        edge_attr='LINK_SENTIMENT',
        create_using=nx.DiGraph() if directed else nx.Graph()
    )
    return G

def compute_louvain(G, undirected=True):
    """
    Compute the Louvain partition of the graph G.
    """
    if undirected:
        G = G.to_undirected()
    # Si edges ont une colonne 'weight', on la normalise
    if 'weight_louvain' not in G.edges[list(G.edges())[0]]:
        for u, v, d in G.edges(data=True):
            d['weight_louvain'] = (d.get('LINK_SENTIMENT', 0) + 1) / 2
    partition = community_louvain.best_partition(G, weight='weight')
    return partition

def count_communities(partition):
    """ Count the number of clusters in the partition."""
    return len(set(partition.values()))

