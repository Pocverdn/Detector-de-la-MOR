import cv2
import numpy as np

from keras.models import load_model
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# Modelo de FaceNet vectorización de resultados
model = load_model('content/facenet_keras.h5', compile=False)
print("Model Loaded Successfully")


def prepared_faces(faces):
    processed_faces = []
    for face in faces:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (160,  160)) #Tamaño para FaceNet()
        face = face.astype("float32") / 255.0 #Normalizar
        face = np.expand_dims(face, axis=0)
        processed_faces.append(face)

    return processed_faces


def visualize_embeddgins(emb):

    all_embeddings = np.vstack([e for e in emb if len(e) > 0]) #Un solo array

    pca = PCA(n_components=2) #Reducir a 2 dimensiones
    reduced = pca.fit_transform(all_embeddings) #Aplicar PCA
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1])
    plt.title('Embeddings de las imágenes (PCA)')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.show()