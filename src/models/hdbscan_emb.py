import numpy as np
import umap
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

def compute_umap_2d(embeddings_norm, random_state=42):
    """Calculate a 2D UMAP embedding for visualization."""
    reducer_2d = umap.UMAP(n_components=2, metric='cosine', random_state=random_state)
    return reducer_2d.fit_transform(embeddings_norm)

def compute_hdbscan(embeddings_norm, use_umap=True, n_components_umap=50, random_state=42, min_cluster_size=10):
    """
    Applied HDBSCAN clustering to the normalized embeddings, with optional UMAP dimensionality reduction.
    Returns:
        labels: np.ndarray, labels HDBSCAN
        silhouette: silhouette score (ignorer outliers)
        n_clusters: nombre de clusters détectés (hors outliers)
    """
    if use_umap:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=n_components_umap, metric='cosine', random_state=random_state)
        embeddings_reduced = reducer.fit_transform(embeddings_norm)
    else:
        embeddings_reduced = embeddings_norm

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    labels = clusterer.fit_predict(embeddings_reduced)

    # silhouette score (ignorer outliers -1)
    mask = labels != -1
    sil = np.nan
    try:
        if mask.sum() >= 2 and len(set(labels[mask])) > 1:
            sil = silhouette_score(embeddings_reduced[mask], labels[mask])
    except Exception:
        sil = np.nan

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, sil, n_clusters