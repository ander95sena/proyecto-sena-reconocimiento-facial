import numpy as np
from configuraciones.config import UMBRAL_SIMILITUD


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
