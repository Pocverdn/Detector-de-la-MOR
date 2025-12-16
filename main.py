import cluster as cluster
import DetectionFaces as detect
import os


def run_pipeline(org_path, dest_path, log=print):

    if not os.listdir(org_path):
        return "No hay fotos disponibles"

    embs, meta = detect.get_embs(org_path, log=log)
    #detect.visual_embs(embs)

    labels_cluster = cluster.clustering(embs)
    cluster.save_clusters(meta, labels_cluster, org_path, dest_path)

    log("Proceso finalizado correctamente")
