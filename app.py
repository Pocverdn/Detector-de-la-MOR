import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras_facenet import FaceNet

# Modelo de reconocimiento de rostros
modelFile = "content/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "content/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Modelo de FaceNet vectorización de resultados
embedder = FaceNet()

def get_detections(photo):

    blob = cv2.dnn.blobFromImage(cv2.resize(photo, (300, 300)), 1, (300,300), (104.0, 177.0, 123.0))

    print("Detectando rostros -- Espere MOR\n")

    net.setInput(blob)

    detections = net.forward()

    return detections

def prepared_faces(faces):
    processed_faces = []
    for face in faces:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (160,  160)) #Tamaño para FaceNet()
        face = face.astype("float32") / 255.0 #Normalizar
        processed_faces.append(face)

    return processed_faces

def get_faces(detections, photo, h, w):
    faces = []
    for i in range(0, detections.shape[2]):
        
        confidence = detections[0, 0, i, 2] #Extrae el nivel de confianza de la detección del rostro

        if confidence > 0.5:

            print("Cara detectada\n")

            box = detections[0,0, i, 3:7] * np.array([w,h, w, h]) # Obtengo las coordenadas x, y del rostro
            (start_x, start_y, end_x, end_y) = box.astype("int")

            face = photo[start_y:end_y, start_x:end_x]
            faces.append(face)

    
    return faces


def load_image():

    directory = "photos"

    for file in os.listdir(directory):

        print(f"Image: {file}")

        photo = cv2.imread(f'photos/{file}')

        (h,  w) = photo.shape[:2]

        detections = get_detections(photo)

        faces_detected = get_faces(detections, photo, h, w)

        embeddgins = prepared_faces(faces_detected)

        results = embedder.embeddings(embeddgins)

        print(f"caras detectadas = {len(results)}\n")

        print(results)


def main():
    load_image()

    


if __name__ == "__main__":
    main()