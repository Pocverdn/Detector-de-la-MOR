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
    dbscan = DBSCAN(eps=0.6, min_samples=3, metric='cosine').fit(embs)
    labels = dbscan.labels_

    return labels


def show_clusters(embs, labels):
    plt.scatter(embs[:,0], embs[:,1], c=labels, cmap="tab10")
    plt.title("Clustering con DBSCAN")
    plt.show()

def save_clusters(labels, labels_cluster):
    os.makedirs('resultados', exist_ok=True)

    for label, label_c in zip(labels, labels_cluster):
        cluster_name = f"cluster_{label_c}" if label_c != -1 else "No encontradas"
        cluster_dir = os.path.join('resultados', cluster_name)
        os.makedirs(cluster_dir, exist_ok=True)

        shutil.copy(os.path.join('photos', label), os.path.join(cluster_dir, label))

def similarity(embs, photo):
    out_vec = np.average([embs[0],embs[1], embs[2], embs[3], embs[4], embs[5], embs[6], embs[7]] , axis=0)

    similarity = cosine_similarity([out_vec],[photo])

    print('Similarity:',similarity)