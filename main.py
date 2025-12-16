import DetectionFaces as detect

import cluster as cluster
import os
import sys

def main():

    path = sys.argv[1]

    if os.listdir(path) == []:
        print("No hay fotos disponibles")
        return None, 0

    embs, meta = detect.get_embs(path)
    
    detect.visual_embs(embs)

    labels_cluster = cluster.clustering(embs)

    #cluster.show_clusters(embs, labels_cluster)

    cluster.save_clusters(meta, labels_cluster, path)

if __name__ == "__main__":
    main()