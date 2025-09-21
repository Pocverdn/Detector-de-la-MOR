import DetectionFaces as detect

import cluster as cluster


def main():
    embs, labels = detect.get_embs("photos")

    #detect.visual_embs(embs)

    labels_cluster = cluster.clustering(embs)

    #cluster.show_clusters(embs, labels_cluster)

    cluster.save_clusters(labels, labels_cluster)

if __name__ == "__main__":
    main()