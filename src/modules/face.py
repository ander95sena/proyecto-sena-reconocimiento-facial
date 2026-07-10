from dataclasses import dataclass
import numpy as np


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
