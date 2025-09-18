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

def main(directory_path):
    emb = []

    for file in os.listdir(directory_path):

        try:

            print(f"Image: {file}")

            photo = cv2.imread(f'{directory_path}/{file}')

            faces = app.get(photo)

            print(f'se detecto: {len(faces)} caras')

            emb.append(faces[0].embedding)

        except ValueError:
            print(f"\nImagen:{file} no analizada\n")

            time.sleep(3)
            continue

    return np.array(emb)

def visual_embs(emb):

    all_embeddings = np.vstack([e for e in emb if len(e) > 0]) #Un solo array

    X_tsne = TSNE(n_components=2, perplexity=15, random_state=42).fit_transform(emb)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne[:,0], X_tsne[:,1])
    plt.title('Embeddings de las imágenes')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.show()

if __name__ == "__main__":
    test = main("photos")

    visual_embs(test)
