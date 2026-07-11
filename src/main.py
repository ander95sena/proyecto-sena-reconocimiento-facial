import cv2 as cv
import numpy as np
import json
from configuraciones.config import (
    PUERTOARDUINO,
    BAUDIOS,
    UMBRAL_SIMILITUD,
    MAX_EMBEDDINGS,
    SKIP_FRAMES,
    P_KALMAN,
    Q_KALMAN,
    R_KALMAN,
    FRAMES_SIN_ROSTRO_PARA_RESET,
    RUTA_JSON,
    RUTA_MODELO,
)
from modules.conexionArduino import crear_conexion_arduino
from modules.messages import Messages
from modules.detector import Detector
from modules.tracker import Tracker
from modules.visualizer import Visualizer
from modules.preprosessing import Preprocessor
from modules.faceRecognition import FaceRecognition
from modules.embeder import FaceNetEmbedder, EmbeddingCollector


if __name__ == "__main__":
    """
        Punto de entrada principal del sistema de verificación de conductor mediante
        reconocimiento facial y comunicación con Arduino.

        Este bloque inicializa todos los componentes necesarios para el proceso:

        - Conexión serial con Arduino (`serialArduino`), con fallback a Dummy si no hay hardware.
        - Detector de rostros (`Detector`) y tracker con filtro de Kalman (`Tracker`).
        - Visualizador (`Visualizer`) para dibujar bounding boxes, landmarks, ojos y puntajes.
        - Mensajero (`Messages`) para mostrar resultados y métricas en pantalla.
        - Preprocesador (`Preprocessor`) y generador de embeddings (`FaceNetEmbedder`).
        - Carga de embeddings de referencia desde `conductor.json`.
        - Reconocedor (`FaceRecognition`) que compara embeddings en vivo contra los registrados.
        - Colector de embeddings (`EmbeddingCollector`) que promedia muestras en varios frames.
        - Captura de video con OpenCV (`cv.VideoCapture`).

        Flujo de ejecución:
        -------------------
        1. Se abre la cámara y se inicializan variables de estado.
        2. En cada frame:
        - Se detecta el rostro principal y se actualiza con el tracker.
        - Se dibujan anotaciones visuales en la imagen.
        - Se muestrean embeddings periódicamente y se acumulan en el colector.
        - Cuando hay suficientes muestras, se calcula el embedding promedio y se verifica.
        - Se muestra en pantalla el resultado (autorizado/no autorizado) y la distancia promedio.
        - Se envía una señal al Arduino (1 = autorizado, 0 = no autorizado).
        - Si no se detecta rostro por varios frames consecutivos, se reinician tracker y colector.
        3. El bucle termina al presionar ESC o 'q'.
        4. Se liberan recursos: cámara, ventanas de OpenCV y conexión serial.

        Variables clave:
        ----------------
        - resultado (str): Mensaje de verificación actual.
        - distancia_promedio (float): Distancia promedio de similitud entre embeddings.
        - señal_arduino (int): Señal enviada al Arduino (0 o 1).
        - frames_sin_rostro (int): Contador de frames sin detección de rostro.

        Este bloque asegura la integración completa entre visión artificial,
        procesamiento biométrico y control electrónico del vehículo.
    """
    arduino = crear_conexion_arduino(PUERTOARDUINO, BAUDIOS)

    detector = Detector()

    tracker = Tracker()

    viz = Visualizer()

    messager = Messages()

    preprocessor = Preprocessor()

    embedder = FaceNetEmbedder(RUTA_MODELO, preprocessor)

    with open(RUTA_JSON, "r") as f:
        embeddings_registro = np.array(json.load(f), dtype=np.float32)

    recognizer = FaceRecognition(embeddings_registro, umbral=UMBRAL_SIMILITUD)

    collector = EmbeddingCollector(
        max_embeddings=MAX_EMBEDDINGS, skip_frames=SKIP_FRAMES
    )

    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara")
        exit()

    resultado = ""

    distancia_promedio = 0.0

    frames_sin_rostro = 0  # Contador de frames sin detección de rostro

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            faces = detector.detect(frame)

            if faces:
                frames_sin_rostro = 0  # Reiniciar contador si se detecta un rostro

                face = detector.get_main_face(faces)

                face = tracker.update(face, P_KALMAN, Q_KALMAN, R_KALMAN)

                viz.draw_bbox(frame, face)

                viz.draw_landmarks(frame, face)

                viz.draw_score(frame, face)

                viz.draw_eyes(frame, face)

                if collector.debe_muestrear():
                    embedding = embedder.get_embedding(frame, face)
                    collector.add(embedding)

                messager.mostrar_contador_muestras(
                    frame, collector.count(), MAX_EMBEDDINGS
                )

                if collector.is_ready():
                    embedding_actual = collector.get_average()

                    autorizado, distancia_promedio = recognizer.verify(embedding_actual)

                    resultado = messager.texto_resultado(autorizado)

                    messager.mostrar_distancia_promedio(frame, distancia_promedio)

                    collector.reset()
            else:
                frames_sin_rostro += 1

                if frames_sin_rostro >= FRAMES_SIN_ROSTRO_PARA_RESET:
                    tracker.reset()
                    collector.reset()
                    resultado = ""
                    distancia_promedio = 0.0
                    frames_sin_rostro = 0
                    arduino.enviar_comando(0,reintentos=2)  # Enviar señal de no autorizado al Arduino

            if resultado:
                messager.mostrar_resultado_verificacion(frame, autorizado)
                messager.mostrar_distancia_promedio(frame, distancia_promedio)

                # Enviar señal al Arduino según el resultado
    
                if resultado == "CONDUCTOR AUTORIZADO":
                    arduino.enviar_comando(1,reintentos=2)  # Enviar señal de autorizado al Arduino
                else:
                    arduino.enviar_comando(0,reintentos=2)  # Enviar señal de no autorizado al Arduino

            cv.imshow("Verificacion Conductor", frame)

            tecla = cv.waitKey(1) & 0xFF

            if tecla == 27 or tecla == ord("q"):
                break

    finally:
        cap.release()

        cv.destroyAllWindows()
        arduino.close()
