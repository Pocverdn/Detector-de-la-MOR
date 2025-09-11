import cv2
import numpy as np
import matplotlib.pyplot as plt

modelFile = "content/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "content/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def detect_face_dnn(img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:  # tune this threshold
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


def load_image():
    img = cv2.imread('photos/_MG_0614.JPG')

    return img

def main():
    img = load_image()

    face = detect_face_dnn(img)

    plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    plt.show()
    
    cv2.imwrite('face.jpg', face)
    

if __name__ == "__main__":
    main()