import cv2
import numpy as np
import matplotlib.pyplot as plt

modelFile = "content/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "content/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def get_detections(photo):

    blob = cv2.dnn.blobFromImage(cv2.resize(photo, (300, 300)), 1, (300,300), (104.0, 177.0, 123.0))

    print("Detectando rostros -- Espere MOR\n")

    net.setInput(blob)

    detections = net.forward()

    return detections

def draw_rectangle(detections, photo, h, w):
    for i in range(0, detections.shape[2]):
        
        confidence = detections[0, 0, i, 2] #Extrae el nivel de confianza de la detección del rostro

        if confidence > 0.5:

            print("Cara detectada")

            box = detections[0,0, i, 3:7] * np.array([w,h, w, h]) # Obtengo las coordenadas x, y del rostro
            (start_x, start_y, end_x, end_y) = box.astype("int")

            cv2.rectangle(photo, (start_x, start_y), (end_x, end_y), (0,225,0),2)

    return photo


def load_image():
    photo = cv2.imread('photos/_MG_0614.JPG')

    return photo

def main():
    photo = load_image()

    (h,  w) = photo.shape[:2]

    detections = get_detections(photo)

    faces_detected = draw_rectangle(detections, photo, h, w)

    plt.imshow(cv2.cvtColor(faces_detected, cv2.COLOR_BGR2RGB))

    cv2.imwrite('face.jpg', faces_detected)

    plt.show()

    


if __name__ == "__main__":
    main()