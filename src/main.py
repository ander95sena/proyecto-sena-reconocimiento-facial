import cv2 as cv
import numpy as np
import json
from modules.config import (
    PUERTOARDUINO,
    BAUDIOS,
    UMBRAL_SIMILITUD,
    MAX_EMBEDDINGS,
    SKIP_FRAMES,
    P_KALMAN,
    Q_KALMAN,
    R_KALMAN,
    FRAMES_SIN_ROSTRO_PARA_RESET,
)
from modules.conexionArduino import crear_conexion_arduino
from modules.messages import Messages

from modules.vision_core import (
    Detector,
    Tracker,
    Visualizer,
    Preprocessor,
    FaceNetEmbedder,
    FaceRecognition,
)


class EmbeddingCollector:
    """
    Administrador de embeddings faciales para procesos de registro o verificación.

    Esta clase se encarga de recolectar embeddings de manera controlada,
    aplicando un salto de frames para evitar redundancia y limitando la
    cantidad máxima de muestras. También provee métodos para calcular
    el embedding promedio y reiniciar la colección.

    Parámetros
    ----------
    max_embeddings : int, opcional
        Número máximo de embeddings a recolectar (por defecto 30).
    skip_frames : int, opcional
        Número de frames a saltar entre cada captura de embedding
        (por defecto 3).

    Atributos
    ---------
    max_embeddings : int
        Límite de embeddings a recolectar.
    skip_frames : int
        Intervalo de frames entre capturas.
    frame_count : int
        Contador de frames procesados.
    embeddings : list
        Lista de embeddings recolectados.

    Métodos
    -------
    count():
        Devuelve el número actual de embeddings recolectados.
    add(embedding: np.ndarray):
        Agrega un embedding a la lista si se cumple la condición de salto
        de frames.
    is_ready():
        Retorna True si se alcanzó el número máximo de embeddings.
    get_average():
        Calcula y devuelve el embedding promedio de los recolectados.
        Si no hay embeddings, retorna None.
    reset():
        Limpia la lista de embeddings y reinicia la colección.

    """

    def __init__(
        self, max_embeddings: int = MAX_EMBEDDINGS, skip_frames: int = SKIP_FRAMES
    ):
        """Inicializa el recolector de embeddings con un límite y un salto de frames."""

        self.max_embeddings = max_embeddings
        self.skip_frames = skip_frames

        self.frame_count = 0

        self.embeddings: list[np.ndarray] = []

    def count(self):
        """Devuelve el número actual de embeddings recolectados."""

        return len(self.embeddings)

    def debe_muestrear(self) -> bool:
        """Incrementa el contador de frames y dice si corresponde tomar una muestra ahora."""
        self.frame_count += 1
        return self.frame_count % self.skip_frames == 0

    def add(self, embedding: np.ndarray):
        """Agrega un embedding a la lista si se cumple la condición de salto de frames."""

        self.embeddings.append(embedding)

    def is_ready(self):
        """Retorna True si se alcanzó el número máximo de embeddings."""

        return len(self.embeddings) >= self.max_embeddings

    def get_average(self):
        """Calcula y devuelve el embedding promedio de los recolectados.
        Si no hay embeddings, retorna None."""

        if not self.embeddings:
            return None

        return np.mean(self.embeddings, axis=0)

    def reset(self):
        """Limpia la lista de embeddings y reinicia la colección."""

        self.embeddings.clear()

        return resultado


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

    embedder = FaceNetEmbedder("faceNet.onnx", preprocessor)

    with open("conductor.json", "r") as f:
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

    señal_arduino = 0  # 0 para no autorizado, 1 para autorizado

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

                    resultado = messager.mostrar_resultado_verificacion(
                        frame, autorizado
                    )

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
                    arduino.write(bytes([0]))

            if resultado:
                messager.mostrar_resultado_verificacion(frame, autorizado)
                messager.mostrar_distancia_promedio(frame, distancia_promedio)

                # Enviar señal al Arduino según el resultado

                if resultado == "CONDUCTOR AUTORIZADO":
                    arduino.write(bytes([1]))
                else:
                    arduino.write(bytes([0]))

            cv.imshow("Verificacion Conductor", frame)

            tecla = cv.waitKey(1) & 0xFF

            if tecla == 27 or tecla == ord("q"):
                break

    finally:
        cap.release()

        cv.destroyAllWindows()
        arduino.close()
