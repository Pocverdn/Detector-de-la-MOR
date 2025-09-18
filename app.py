import cv2
import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import ProcessFaces as process
import DetectFaces as detect

from numpy import linalg as LA

def load_images(directory_path):

    emb = []

    for file in os.listdir(directory_path):

        try:

            print(f"Image: {file}")

            photo = cv2.imread(f'photos/{file}')

            (h,  w) = photo.shape[:2]

            detection = detect.get_detections(photo)

            faces_detected = detect.get_faces(detection, photo, h, w)

            faces = process.prepared_faces(faces_detected)

            if len(faces) > 1: #Mas de una caras
                for face in faces:
                    embeddings = process.model.predict(face)
                    embeddings_norm = embeddings / LA.norm(embeddings) #Normalizo el vector
                    print(f'Normalizada: {LA.norm(embeddings_norm)}\n')
                    emb.append(embeddings_norm)
            else:
                embeddings = process.model.predict(faces)
                embeddings_norm = embeddings / LA.norm(embeddings)
                print(f'Normalizada: {LA.norm(embeddings_norm)}\n')
                emb.append(embeddings_norm)

            print(f"caras detectadas = {len(embeddings)}\n")

        except ValueError:
            print(f"\nImagen:{file} no analizada\n")

            time.sleep(3)
            continue

    return emb

def main():
    emb = load_images("photos")

    process.visual_embs(emb)
    
if __name__ == "__main__":
    main()