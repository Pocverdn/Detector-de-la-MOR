import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.manifold import TSNE


def clustering(embs):
    dbscan = DBSCAN(eps=0.35, min_samples=3, metric='cosine').fit(embs)
    labels = dbscan.labels_

    print(labels)

    return labels


def show_clusters(embs, labels):
    X_tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    X_2d = X_tsne.fit_transform(embs)

    plt.figure(figsize=(10,6))
    plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap="tab20")
    plt.title("Clustering con DBSCAN (TSNE)")
    plt.show()

def save_clusters(meta, labels_cluster, photos_path):
    os.makedirs('resultados', exist_ok=True)

    for m, cluster_id in zip(meta, labels_cluster):
        cluster_name = f"cluster_{cluster_id}" if cluster_id != -1 else "No_encontradas"
        cluster_dir = os.path.join('resultados', cluster_name)
        os.makedirs(cluster_dir, exist_ok=True)

        src = os.path.join(photos_path, m["file"])
        dst = os.path.join(cluster_dir, m["file"])

        if not os.path.exists(dst):
            shutil.copy(src, dst)
