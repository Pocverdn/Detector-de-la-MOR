import DetectionFaces as detect

import cluster as cluster


def main():
    embs = detect.get_embs("photos")

    detect.visual_embs(embs)

    clusters = cluster.clustering(embs)

    cluster.visualize(embs, clusters)

if __name__ == "__main__":
    main()