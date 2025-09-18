# Detector de la MOR 

Este es un programa diseñado para realizar automaticamente el proceso de encarpetado para fotos de graduaciones.

## IMPORTANTE:
Se recomienda hacer uso de Anaconda para crear un entorno virtual con las siguientes especificaciones

Lenguaje: Python3.6.2

Librerias: 
  * matplotlib 3.3.4
  * Numpy 1.19.5
  * opencv-python 4.5.5.64
  * scikit-learn 0.24.2
  * Keras 2.1.2

Modelos:
* res10_300x300_ssd_iter_140000.caffemodel: Modelo pre-entrenado para la detección de rostros
* facenet_keras.h5: Modelo pre-entreando para el reconocimiento facial
