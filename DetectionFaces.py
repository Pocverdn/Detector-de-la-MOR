import cv2
import time
import os
import insightface
import numpy as np
from insightface.app import FaceAnalysis

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']) 
app.prepare(ctx_id=-1) #-1 para CPU

def get_embs(path, log=print):
    embs = []
    meta = []

    detect = 0
    no_detect = 0

    for file in os.listdir(path):

        log(f"Image: {file}")

        photo = cv2.imread(f'{path}/{file}')
        faces = app.get(photo)

        log(f"Se detectó: {len(faces)} caras")


        if len(faces) >= 1:
            for i, face in enumerate(faces):
                embs.append(face.normed_embedding)
                meta.append({"file": file, "face_id": i})
        else:
            log(f"Imagen {file} no analizada")

            no_detect += 1

            time.sleep(3)
            continue

        detect += 1


    log(f"Detectadas: {detect} — No detectadas: {no_detect}")

    return np.array(embs), meta

def get_photo_embs(photo):
    img = cv2.imread(photo)

    face = app.get(img)

    return face[0].normed_embedding

def visual_embs(emb):

    X_tsne = TSNE(n_components=2, perplexity=15, random_state=42).fit_transform(emb)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne[:,0], X_tsne[:,1])
    plt.title('Embeddings de las imágenes')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.show()