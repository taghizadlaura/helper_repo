import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from kneed import KneeLocator
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import silhouette_score

def normalize_emb(X: np.ndarray) -> np.ndarray:
    """Normalize the embeddings using L2 normalization."""
    return normalize(X, norm='l2')

def find_best_k(X: np.ndarray, k_values=None, batch_size: int = 1024, random_state: int = 42) -> int:
    """Find best k using the elbow method on KMeans clustering."""
    if k_values is None:
        k_values = [10, 50, 100, 200, 300, 400, 500, 600]

    inertias = []
    for k in tqdm(k_values):
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=random_state)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    kl = KneeLocator(k_values, inertias, curve="convex", direction="decreasing")
    best_k = kl.elbow
    print(f"Best k estimated : {best_k}")

    # Plot du coude
    plt.figure(figsize=(8,5))
    plt.plot(k_values, inertias, 'o-', linewidth=2)
    if best_k is not None:
        plt.axvline(best_k, color='r', linestyle='--', label=f'Estimated elbow: {best_k}')
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia intracluster")
    plt.grid(True)
    plt.legend()
    plt.show()

    return best_k

def compute_kmeans_clusters(X: np.ndarray, n_clusters: int, batch_size: int = 1024, random_state: int = 42) -> np.ndarray:
    """Applied KMeans clustering to array and return cluster labels."""
    kmeans_final = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state)
    kmeans_final.fit(X)
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X, kmeans_final.labels_)
    print(f"Silhouette Score: {silhouette_avg}")
    return kmeans_final.labels_

def compute_tsne(X: np.ndarray, n_components=2, perplexity=30, random_state=42) -> np.ndarray:
    """Compute t-SNE reduction of the embeddings."""
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    return tsne.fit_transform(X)