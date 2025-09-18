import cv2
import numpy as np

# Modelo de reconocimiento de rostros
modelFile = "content/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "content/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def get_detections(photo):

    blob = cv2.dnn.blobFromImage(cv2.resize(photo, (300, 300)), 1, (300,300), (104.0, 177.0, 123.0))

    print("Detectando rostros -- Espere MOR\n")

    net.setInput(blob)

    detections = net.forward()

    return detections


def get_faces(detections, photo, h, w):
    faces = []
    for i in range(0, detections.shape[2]):
        
        confidence = detections[0, 0, i, 2] #Extrae el nivel de confianza de la detección del rostro

        if confidence > 0.6:

            print("Cara detectada\n")

            box = detections[0,0, i, 3:7] * np.array([w,h, w, h]) # Obtengo las coordenadas x, y del rostro
            (start_x, start_y, end_x, end_y) = box.astype("int")

            face = photo[start_y:end_y, start_x:end_x]
            faces.append(face)

    
    return faces