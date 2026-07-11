import numpy as np
from modules.face import Face
from modules.preprosessing import Preprocessor
import onnxruntime as ort
from configuraciones.config import MAX_EMBEDDINGS, SKIP_FRAMES
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("embedder")


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
        logger.info(f"FaceNetEmbedder inicializado con modelo: {model_path}")

    def get_embedding(self, frame: np.ndarray, face: Face):
        """Obtiene el embedding de un rostro en un fotograma, aplicando alineación
        y preprocesamiento antes de la inferencia."""

        # Usar el preprocessor externo
        aligned_face = self.preprocessor.align_face(frame, face)
        preprocessed = self.preprocessor.preprocess(aligned_face)
        embedding = self.session.run(None, {self.input_name: preprocessed})[0]
        logger.info("Embedding obtenido con éxito.")
        return embedding[0]

    def add_embedding(self, frame: np.ndarray, face: Face):
        """Calcula el embedding de un rostro y lo guarda en el buffer interno."""

        emb = self.get_embedding(frame, face)
        self.embeddings_buffer.append(emb)
        logger.info("Embedding agregado al buffer.")

    def get_average_embedding(self):
        """Devuelve el embedding promedio de todos los almacenados en el buffer.
        Si no hay embeddings, retorna None."""

        if not self.embeddings_buffer:
            return None
        logger.info(
            f"Calculando embedding promedio de {len(self.embeddings_buffer)} embeddings."
        )
        return np.mean(self.embeddings_buffer, axis=0)


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
        logger.info(
            f"EmbeddingCollector inicializado con max_embeddings={max_embeddings}, skip_frames={skip_frames}"
        )

    def count(self):
        """Devuelve el número actual de embeddings recolectados."""

        logger.info(f"Conteo de embeddings: {len(self.embeddings)}")
        return len(self.embeddings)

    def debe_muestrear(self) -> bool:
        """Incrementa el contador de frames y dice si corresponde tomar una muestra ahora."""
        self.frame_count += 1
        logger.info(
            f"Frame count incrementado a {self.frame_count}. Debe muestrear: {self.frame_count % self.skip_frames == 0}"
        )
        return self.frame_count % self.skip_frames == 0

    def add(self, embedding: np.ndarray):
        """Agrega un embedding a la lista si se cumple la condición de salto de frames."""
        logger.info("Agregando embedding al buffer.")
        self.embeddings.append(embedding)

    def is_ready(self):
        """Retorna True si se alcanzó el número máximo de embeddings."""

        logger.info(
            f"Verificando si el recolector está listo: {len(self.embeddings) >= self.max_embeddings}"
        )
        return len(self.embeddings) >= self.max_embeddings

    def get_average(self):
        """Calcula y devuelve el embedding promedio de los recolectados.
        Si no hay embeddings, retorna None."""

        if not self.embeddings:
            return None
        logger.info(
            f"Calculando embedding promedio de {len(self.embeddings)} embeddings."
        )
        return np.mean(self.embeddings, axis=0)

    def reset(self):
        """Limpia la lista de embeddings y reinicia la colección."""
        logger.info("Reinicio de  embedding collector.")

        self.embeddings.clear()
