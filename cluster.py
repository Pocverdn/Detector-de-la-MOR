import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from sklearn.manifold import TSNE


def clustering(embs):
    dbscan = DBSCAN(eps=0.3, min_samples=8)
    clusters = dbscan.fit_predict(embs)

    return clusters

