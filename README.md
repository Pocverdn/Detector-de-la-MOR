# Detector de la MOR 

Este es un programa diseñado para realizar automaticamente el proceso de encarpetado para fotos de graduaciones.

## IMPORTANTE:
Este proyecto incluye versiones antiguas del detector con diferentes herramientas que estuve utilizando mientras aprendia, a continuación se muestran las especificaciones de las versiones disponibles hasta la fecha:

### Detector 2.0

Lenguaje: Python3.13.1

Librerias:
 * opencv-python 4.12.0.88
 * numpy 2.2.2
 * insightface 0.7.3
 * scikit-learn 1.7.2
 * matplotlib 3.10.0

Modelo:
 * buffalo_l: Modelo de ArcFace para deteción y procesamiento de rostros, se puede ejecutar por CPU Y GPU

### Detector 1.0

Para trabajar con detector1.0 se recomienda hacer uso de Anaconda creando un entorno virtual con las siguientes especificaciones

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
