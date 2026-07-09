from dataclasses import dataclass
import cv2 as cv
import numpy as np
import json
from insightface.app import FaceAnalysis
from filterpy.kalman import KalmanFilter
import onnxruntime as ort
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
)


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
    def __init__(self, puerto: str=PUERTOARDUINO, baudrate: int=BAUDIOS):
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


@dataclass
class Face:
    """
    Representa una detección facial con su caja delimitadora, puntos de referencia
    (landmarks) y nivel de confianza.

    Esta clase facilita el acceso a las coordenadas de la cara detectada y provee
    propiedades útiles como el centro y dimensiones de la caja.

    Parámetros
    ----------
    bbox : array-like
        Coordenadas de la caja delimitadora en formato [x1, y1, x2, y2].
    landmarks : array-like
        Puntos clave del rostro (ojos, nariz, boca, etc.), si están disponibles.
    score : float
        Confianza de la detección facial (0.0–1.0).

    Atributos
    ---------
    bbox : np.ndarray
        Caja delimitadora del rostro.
    kps : np.ndarray
        Puntos clave del rostro.
    score : float
        Nivel de confianza de la detección.

    Propiedades
    -----------
    x : int
        Coordenada X inicial de la caja.
    y : int
        Coordenada Y inicial de la caja.
    w : int
        Ancho de la caja delimitadora.
    h : int
        Alto de la caja delimitadora.
    cx : int
        Coordenada X del centro de la caja.
    cy : int
        Coordenada Y del centro de la caja.
    """

    def __init__(self, bbox: np.ndarray, landmarks: np.ndarray, score: float):
        """Inicializa un objeto Face con la caja delimitadora, landmarks y score."""

        self.bbox = bbox
        self.kps = landmarks
        self.score = score

    @property
    def x(self):
        return int(self.bbox[0])

    @property
    def y(self):
        return int(self.bbox[1])

    @property
    def w(self):
        return int(self.bbox[2] - self.bbox[0])

    @property
    def h(self):
        return int(self.bbox[3] - self.bbox[1])

    @property
    def cx(self):
        return int(self.x + self.w / 2)

    @property
    def cy(self):
        return int(self.y + self.h / 2)


class Detector:
    """
    Detector de rostros basado en la librería InsightFace.

    Esta clase inicializa el motor de análisis facial y permite detectar
    múltiples rostros en un fotograma, extrayendo su caja delimitadora,
    puntos de referencia (landmarks) y nivel de confianza. También incluye
    un método para seleccionar el rostro principal en la escena.

    Métodos
    -------
    detect(frame):
        Procesa un fotograma (imagen) y devuelve una lista de objetos `Face`
        con la información de cada rostro detectado.
    get_main_face(faces):
        Retorna el rostro principal de la lista de detecciones, definido como
        el de mayor área (ancho × alto).

    """

    def __init__(self):
        """Inicializa el detector de rostros usando InsightFace con soporte para CPU."""

        # Nota: Asegúrate de que tu entorno soporte 'DmlExecutionProvider' (Windows/DirectX)
        self.app = FaceAnalysis(
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "landmark_2d_106"],
        )
        self.app.prepare(ctx_id=0, det_size=(320, 320))

    def detect(self, frame: np.ndarray):
        """Detecta rostros en un fotograma y devuelve una lista de objetos `Face`."""

        faces_raw = self.app.get(frame)
        faces = []
        for f in faces_raw:
            landmarks = f.landmark_2d_106 if f.landmark_2d_106 is not None else f.kps
            faces.append(Face(f.bbox, landmarks, f.det_score))
        return faces

    def get_main_face(self, faces: list[Face]):
        """Selecciona el rostro principal de la lista, definido como el de mayor área."""

        face = max(faces, key=lambda f: f.w * f.h)
        return face


class Tracker:
    """
    Rastreador de rostros usando un Filtro de Kalman.

    Esta clase implementa un modelo de velocidad constante para suavizar
    las coordenadas de la detección facial en cada fotograma. El estado
    incluye posición, tamaño y velocidades en X e Y, lo que permite
    predecir el movimiento del rostro entre frames.

    Métodos
    -------
    init_filter(face):
        Inicializa el filtro de Kalman con el estado inicial del rostro
        detectado. El estado es [x, y, w, h, vx, vy].
    update(face):
        Actualiza el filtro con una nueva medición [x, y, w, h], predice
        el estado y devuelve el objeto `Face` con su caja delimitadora
        suavizada.

    Detalles técnicos
    -----------------
    - Estado (dim_x = 6): [x, y, w, h, vx, vy]
    - Medición (dim_z = 4): [x, y, w, h]
    - Matriz de transición (F): modelo de velocidad constante
    - Matriz de medición (H): mapea estado → medición
    - P: incertidumbre inicial (100.0)
    - R: ruido de medición (2.0)
    - Q: ruido del proceso (0.1)

    """

    def __init__(self):
        self.kf = None

    def init_filter(self, face: Face, P: float = 100.0, R: float = 2.0, Q: float = 0.1):
        """Inicializa el filtro de Kalman con el estado inicial del rostro detectado."""

        # Estado: [x, y, w, h, vx, vy] -> Agregamos velocidades vx, vy
        estado = np.array([face.x, face.y, face.w, face.h, 0, 0], dtype=np.float32)

        # dim_x = 6 (estado), dim_z = 4 (medición: x, y, w, h)
        self.kf = KalmanFilter(dim_x=6, dim_z=4)
        self.kf.x = estado

        # Matriz de transición de estado (F) - Modelo de velocidad constante
        # x_new = x + vx * dt (asumiendo dt = 1 frame)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0],  # x
                [0, 1, 0, 0, 0, 1],  # y
                [
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                ],  # w (asumimos que el tamaño no cambia por velocidad)
                [0, 0, 0, 1, 0, 0],  # h
                [0, 0, 0, 0, 1, 0],  # vx
                [0, 0, 0, 0, 0, 1],  # vy
            ]
        )

        # Matriz de medición (H): mapea el estado de 6 dimensiones a las 4 que medimos
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]
        )

        self.kf.P *= P  # Incertidumbre inicial
        self.kf.R = np.eye(4) * R  # Ruido de la medición (confianza en el detector)
        self.kf.Q = np.eye(6) * Q  # Ruido del proceso (dinámica del sistema)

    def update(self, face: Face, P: float, Q: float, R: float):
        """Actualiza el filtro con una nueva medición y devuelve el rostro actualizado."""
        if self.kf is None:
            self.init_filter(face, P, Q, R)
            return face

        medida = np.array([face.x, face.y, face.w, face.h], dtype=np.float32)

        self.kf.predict()
        self.kf.update(medida)

        # Extraemos solo los 4 primeros elementos (x, y, w, h)
        x, y, w, h = self.kf.x[:4].astype(int)

        # Actualizamos el objeto Face original con la predicción filtrada
        face.bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
        return face


class Visualizer:
    """
    Clase para dibujar elementos gráficos sobre un fotograma usando OpenCV.

    Esta clase provee métodos para visualizar la información asociada a un
    objeto `Face`: caja delimitadora, puntos de referencia, nivel de confianza
    y regiones de los ojos. Es útil para depuración y representación visual
    en sistemas de visión artificial.

    Métodos
    -------
    draw_bbox(frame, face):
        Dibuja la caja delimitadora del rostro en color verde.
    draw_landmarks(frame, face):
        Dibuja los puntos clave (landmarks) del rostro en color azul.
    draw_score(frame, face):
        Muestra el puntaje de confianza de la detección sobre el rostro.
    draw_eyes(frame, face):
        Dibuja rectángulos alrededor de los ojos usando landmarks específicos
        (38 y 88 en el modelo de 106 puntos). El tamaño del rectángulo se
        adapta proporcionalmente al ancho de la cara.

    """

    def draw_bbox(self, frame: np.ndarray, face: Face):
        """Dibuja la caja delimitadora del rostro en color verde sobre el fotograma."""

        x1, y1, x2, y2 = face.bbox.astype(int)
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def draw_landmarks(self, frame: np.ndarray, face: Face):
        """Dibuja los puntos clave (landmarks) del rostro en color azul sobre el fotograma."""

        if face.kps is not None:
            for lx, ly in face.kps.astype(int):
                cv.circle(frame, (lx, ly), 2, (255, 0, 0), -1)

    def draw_score(self, frame: np.ndarray, face: Face):
        """Muestra el puntaje de confianza de la detección sobre el rostro."""

        cv.putText(
            frame,
            f"Score:{face.score:.2f}",
            (face.x, face.y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    def draw_eyes(self, frame: np.ndarray, face: Face):
        """Dibuja rectángulos alrededor de los ojos usando landmarks específicos."""

        if face.kps is None or len(face.kps) < 90:
            return

        # Usar landmarks 106
        eye_left = tuple(map(int, face.kps[38]))
        eye_right = tuple(map(int, face.kps[88]))

        # Tamaño adaptativo según el ancho de la cara
        tam = max(10, int(face.w * 0.1))

        for ex, ey in [eye_left, eye_right]:
            cv.rectangle(
                frame, (ex - tam, ey - tam), (ex + tam, ey + tam), (255, 0, 0), 2
            )


class Preprocessor:
    """
    Clase para alinear y preprocesar rostros antes de generar embeddings.

    Esta clase implementa dos pasos clave en el pipeline de reconocimiento facial:
    1. **Alineación**: usa landmarks (ojos y boca) para aplicar una transformación
    afín que normaliza la posición del rostro en una imagen estándar de 160×160.
    2. **Preprocesamiento**: convierte la imagen a RGB, normaliza los valores
    de píxel al rango [-1, 1] y agrega una dimensión extra para compatibilidad
    con modelos de deep learning (ej. FaceNet).

    Métodos
    -------
    align_face(frame, face):
        Alinea el rostro detectado usando landmarks clave (ojo izquierdo, ojo derecho,
        boca) y devuelve la cara transformada en un tamaño fijo de 160×160 píxeles.
    preprocess(face_img):
        Convierte la imagen a RGB, normaliza los valores de píxel y devuelve un
        tensor listo para ser usado como entrada en un modelo de embeddings.
    """

    def align_face(self, frame: np.ndarray, face: Face):
        """Alinea el rostro detectado usando landmarks clave y devuelve la cara transformada."""

        # Seleccionamos landmarks clave: ojo izquierdo, ojo derecho y boca
        left_eye = face.kps[38]  # índice aproximado ojo izquierdo
        right_eye = face.kps[88]  # índice aproximado ojo derecho
        mouth = face.kps[66]  # índice aproximado centro boca

        # Puntos de referencia actuales
        src_points = np.array([left_eye, right_eye, mouth], dtype=np.float32)

        # Puntos de referencia destino (posición estándar)
        dst_points = np.array(
            [
                [50.0, 50.0],  # ojo izquierdo
                [110.0, 50.0],  # ojo derecho
                [80.0, 100.0],  # boca
            ],
            dtype=np.float32,
        )

        # Calcular transformación afín
        M = cv.getAffineTransform(src_points, dst_points)

        # Aplicar transformación
        aligned = cv.warpAffine(frame, M, (160, 160))

        return aligned

    def preprocess(self, face_img: np.ndarray):
        """Convierte la imagen a RGB, normaliza los valores de píxel y devuelve un tensor listo para el modelo."""
        face_rgb = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype(np.float32) / 127.5 - 1.0
        return np.expand_dims(face_normalized, axis=0)


class FaceNetEmbedder:
    """
    Generador de embeddings faciales usando FaceNet (ONNX).

    Esta clase encapsula la carga del modelo FaceNet y el proceso de
    alineación + preprocesamiento de rostros para obtener vectores de
    características (embeddings). También permite acumular múltiples
    embeddings y calcular un promedio robusto.

    Parámetros
    ----------
    model_path : str, opcional
        Ruta al modelo ONNX de FaceNet. Por defecto "faceNet.onnx".
    preprocessor : Preprocessor, opcional
        Objeto encargado de la alineación y normalización de rostros.
        Si no se pasa, se crea uno por defecto.

    Atributos
    ---------
    session : onnxruntime.InferenceSession
        Sesión de inferencia para ejecutar el modelo FaceNet.
    input_name : str
        Nombre de la entrada del modelo ONNX.
    embeddings_buffer : list
        Lista de embeddings acumulados para cálculo posterior.
    preprocessor : Preprocessor
        Instancia usada para alinear y normalizar rostros.

    Métodos
    -------
    get_embedding(frame, face):
        Obtiene el embedding de un rostro en un fotograma, aplicando
        alineación y preprocesamiento antes de la inferencia.
    add_embedding(frame, face):
        Calcula el embedding de un rostro y lo guarda en el buffer interno.
    get_average_embedding():
        Devuelve el embedding promedio de todos los almacenados en el buffer.
        Si no hay embeddings, retorna None.

    """

    def __init__(self, model_path="faceNet.onnx", preprocessor=None):
        """Inicializa el generador de embeddings cargando el modelo FaceNet y configurando el preprocesador."""

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.embeddings_buffer = []
        # Si no pasas un preprocessor, crea uno por defecto
        self.preprocessor = preprocessor if preprocessor else Preprocessor()

    def get_embedding(self, frame: np.ndarray, face: Face):
        """Obtiene el embedding de un rostro en un fotograma, aplicando alineación
        y preprocesamiento antes de la inferencia."""

        # Usar el preprocessor externo
        aligned_face = self.preprocessor.align_face(frame, face)
        preprocessed = self.preprocessor.preprocess(aligned_face)
        embedding = self.session.run(None, {self.input_name: preprocessed})[0]
        return embedding[0]

    def add_embedding(self, frame: np.ndarray, face: Face):
        """Calcula el embedding de un rostro y lo guarda en el buffer interno."""

        emb = self.get_embedding(frame, face)
        self.embeddings_buffer.append(emb)

    def get_average_embedding(self):
        """Devuelve el embedding promedio de todos los almacenados en el buffer.
        Si no hay embeddings, retorna None."""

        if not self.embeddings_buffer:
            return None
        return np.mean(self.embeddings_buffer, axis=0)


class FaceRecognition:
    """
    Sistema de verificación facial basado en embeddings.

    Esta clase compara un embedding facial actual contra un conjunto de
    embeddings de registro (previamente almacenados) usando distancia
    euclidiana. El resultado indica si el conductor está autorizado o no,
    según un umbral definido.

    Parámetros
    ----------
    embeddings_registro : np.ndarray
        Lista de embeddings faciales de referencia (registro autorizado).
    umbral : float, opcional
        Valor límite de distancia promedio para considerar al conductor
        como autorizado. Por defecto 0.7.

    Métodos
    -------
    normalize(embedding):
        Normaliza un embedding dividiéndolo por su norma L2.
    euclidian_distance(a, b):
        Calcula la distancia euclidiana entre dos embeddings.
    verify(embedding_actual):
        Normaliza el embedding actual, calcula las distancias contra todos
        los embeddings de registro y devuelve:
        - autorizado (bool): True si la distancia promedio ≤ umbral.
        - distancia_promedio (float): valor de la distancia promedio.

    """

    def __init__(
        self, embeddings_registro: np.ndarray, umbral: float = UMBRAL_SIMILITUD
    ):
        """Inicializa el sistema de verificación con embeddings de registro y un umbral de distancia."""

        self.embeddings_registro = embeddings_registro
        self.umbral = umbral

    def normalize(self, embedding: np.ndarray):
        """Normaliza un embedding dividiéndolo por su norma L2."""

        return embedding / np.linalg.norm(embedding)

    def euclidian_distance(self, a: np.ndarray, b: np.ndarray):
        """Calcula la distancia euclidiana entre dos embeddings."""

        return float(np.linalg.norm(a - b))

    def verify(self, embedding_actual: np.ndarray):
        """Normaliza el embedding actual, calcula las distancias contra todos
        los embeddings de registro y devuelve:
         el resultado de autorización y la distancia promedio."""

        embedding_actual = self.normalize(embedding_actual)

        distances = [
            self.euclidian_distance(embedding_actual, emb)
            for emb in self.embeddings_registro
        ]

        distancia_promedio = np.mean(distances)

        autorizado = distancia_promedio <= self.umbral

        return autorizado, distancia_promedio


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

    def add(self, embedding: np.ndarray):
        """Agrega un embedding a la lista si se cumple la condición de salto de frames."""

        self.frame_count += 1

        if self.frame_count % self.skip_frames != 0:
            return

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
    MensajeResultado():
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

    def MensajeResultado(self):
        """Devuelve el texto del resultado de verificación según la variable `autorizado`."""

        resultado = "CONDUCTOR AUTORIZADO" if autorizado else "CONDUCTOR NO AUTORIZADO"
        return resultado


if __name__ == "__main__":
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

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            faces = detector.detect(frame)

            if faces:
                face = detector.get_main_face(faces)

                face = tracker.update(face, P_KALMAN, Q_KALMAN, R_KALMAN)

                viz.draw_bbox(frame, face)

                viz.draw_landmarks(frame, face)

                viz.draw_score(frame, face)

                viz.draw_eyes(frame, face)

                embedding = embedder.get_embedding(frame, face)

                embedding = recognizer.normalize(embedding)

                collector.add(embedding)

                messager.Mensajecontador_muestras(frame, collector)

                if collector.is_ready():
                    embedding_actual = collector.get_average()

                    autorizado, distancia_promedio = recognizer.verify(embedding_actual)

                    resultado = messager.MensajeResultado()

                    collector.reset()

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
