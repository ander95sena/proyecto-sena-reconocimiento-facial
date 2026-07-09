import cv2 as cv
import numpy as np
import json
import serial
import time
import serial.tools.list_ports
from abc import ABC, abstractmethod
from config import (
    PUERTOARDUINO,
    BAUDIOS,
    UMBRAL_SIMILITUD,
    MAX_EMBEDDINGS,
    SKIP_FRAMES,
    P_KALMAN,
    Q_KALMAN,
    R_KALMAN,
    FRAMES_SIN_ROSTRO_PARA_RESET
)

from vision_core import Face, Detector, Tracker, Visualizer, Preprocessor, FaceNetEmbedder, normalize,FaceRecognition


class BaseSerial(ABC):
    """
    Clase base abstracta para definir la interfaz de comunicación serial.

    Esta clase establece los métodos que cualquier implementación de conexión
    serial debe proporcionar, permitiendo la flexibilidad de usar tanto
    conexiones reales como simuladas (DummySerial) sin cambiar la lógica
    del programa principal.

    Métodos abstractos:
        write(data): Envía datos a través de la conexión serial.
        close(): Cierra la conexión serial.
    """

    def __init__(self, puerto: str = PUERTOARDUINO, baudrate: int = BAUDIOS):
        self.port = puerto
        self.baudrate = baudrate
        self.is_open = True

    @abstractmethod
    def write(self, data: bytes) -> None:
        """Envía datos por la conexión serial."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Cierra la conexión serial."""
        pass


class DummySerial(BaseSerial):
    """
    Implementación simulada de una conexión serial.
    """

    def __init__(self, puerto: str, baudrate: int):
        super().__init__(puerto, baudrate)
        print(
            f"⚠️ No se pudo conectar al puerto {puerto}. Usando conexión simulada (Dummy)."
        )

    def write(self, data: bytes) -> None:
        print(f"[Dummy] Enviando datos simulados: {list(data)}")

    def close(self) -> None:
        self.is_open = False
        print("[Dummy] Conexión simulada cerrada")


class serialArduino(BaseSerial):
    """
    Implementación real de una conexión serial con Arduino.
    """

    def __init__(self, puerto: str = PUERTOARDUINO, baudrate: int = BAUDIOS):
        super().__init__(puerto, baudrate)
        try:
            puertos = [p.device for p in serial.tools.list_ports.comports()]
            if puerto not in puertos:
                raise serial.SerialException(f"Puerto {puerto} no encontrado")

            self.conexion = serial.Serial(puerto, baudrate)
            time.sleep(2)
            print(f"✅ Conectado a {puerto} a {baudrate} baudios")
        except serial.SerialException as e:
            print(f"⚠️ Error de conectividad: {e}")
            print("➡️ Activando conexión Dummy para pruebas de software.")
            self.conexion = DummySerial(puerto, baudrate)

    # Métodos abstractos obligatorios de BaseSerial
    def write(self, data: bytes) -> None:
        if self.conexion and self.conexion.is_open:
            self.conexion.write(data)
        else:
            print("⚠️ No hay conexión abierta")

    def close(self) -> None:
        if self.conexion and self.conexion.is_open:
            self.conexion.close()
            print("🔌 Conexión cerrada")

    # Métodos compatibles con tu código anterior
    def iniciar(self) -> None:
        print("🔌 Conexión iniciada (ya establecida en __init__)")

    def enviarSeñal(self, dato: int) -> None:
        self.write(bytes([dato]))





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


class Messages:
    """
    Clase para mostrar mensajes informativos en pantalla durante el proceso
    de registro y verificación del conductor.

    Esta clase encapsula la lógica de visualización con OpenCV (`cv.putText`)
    para mostrar el conteo de muestras, resultados de verificación y métricas
    de distancia/similitud. Facilita la depuración y la interacción visual
    con el sistema de reconocimiento facial.

    Métodos
    -------
    Mensajecontador_muestras(frame: np.ndarray, collector: EmbeddingCollector):
        Muestra en pantalla el número de muestras recolectadas sobre el total (30).
    Mensajeresultado_verificacion(frame: np.ndarray, resultado: str):
        Muestra el resultado de la verificación ("AUTORIZADO" o "NO AUTORIZADO")
        en color verde o rojo según corresponda.
    Mensajedistancia_promedio(frame: np.ndarray, distancia_promedio: float):
        Muestra la distancia promedio (o similitud) calculada contra los embeddings
        de referencia, con tres decimales.
    MensajeResultado(autorizado: bool):
        Devuelve el texto del resultado de verificación según la variable `autorizado`.

    """

    def Mensajecontador_muestras(
        self, frame: np.ndarray, collector: EmbeddingCollector
    ):
        """Muestra en pantalla el número de muestras recolectadas sobre el total (30)."""

        cv.putText(
            frame,
            f"Muestras: {collector.count()}/{MAX_EMBEDDINGS}",
            (20, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    def Mensajeresultado_verificacion(self, frame: np.ndarray, resultado: str):
        """Muestra el resultado de la verificación ("AUTORIZADO" o "NO AUTORIZADO")
        en color verde o rojo según corresponda."""

        cv.putText(
            frame,
            resultado,
            (20, 80),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if "AUTORIZADO" in resultado else (0, 0, 255),
            2,
        )

    def Mensajedistancia_promedio(self, frame: np.ndarray, distancia_promedio: float):
        """Muestra la distancia promedio (o similitud) calculada contra los embeddings
        de referencia, con tres decimales."""

        cv.putText(
            frame,
            f"DIST: {distancia_promedio:.3f}",
            (20, 120),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

    def MensajeResultado(self,autorizado: bool):
        """Devuelve el texto del resultado de verificación según la variable `autorizado`."""

        resultado = "CONDUCTOR AUTORIZADO" if autorizado else "CONDUCTOR NO AUTORIZADO"
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


    arduino = serialArduino()

    arduino.iniciar()

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

    frames_sin_rostro = 0 # Contador de frames sin detección de rostro

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

                messager.Mensajecontador_muestras(frame, collector)


                if collector.is_ready():
                    embedding_actual = collector.get_average()

                    autorizado, distancia_promedio = recognizer.verify(embedding_actual)

                    resultado = messager.MensajeResultado(autorizado)

                    collector.reset()
            else:
                frames_sin_rostro += 1

                if frames_sin_rostro >= FRAMES_SIN_ROSTRO_PARA_RESET:
                    tracker.reset()
                    collector.reset()
                    resultado = ""
                    distancia_promedio = 0.0
                    frames_sin_rostro = 0
                    arduino.enviarSeñal(0)


            if resultado:
                messager.Mensajeresultado_verificacion(frame, resultado)

                messager.Mensajedistancia_promedio(frame, distancia_promedio)

                # Enviar señal al Arduino según el resultado

                if resultado == "CONDUCTOR AUTORIZADO":
                    arduino.enviarSeñal(1)
                else:
                    arduino.enviarSeñal(0)

            cv.imshow("Verificacion Conductor", frame)

            tecla = cv.waitKey(1) & 0xFF

            if tecla == 27 or tecla == ord("q"):
                break

    finally:
        cap.release()

        cv.destroyAllWindows()
        arduino.close()
