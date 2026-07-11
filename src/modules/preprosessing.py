import numpy as np
import cv2 as cv
from modules.face import Face
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("preprocessor")


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

    def mejorar_iluminacion(self, face_img: np.ndarray) -> np.ndarray:
        """
        Normaliza el contraste local de la imagen usando CLAHE sobre el canal
        de luminancia (espacio LAB), para reducir el efecto de iluminación
        desigual (p. ej. luz lateral de ventana en la cabina de un vehículo)
        sin distorsionar el balance de color.
        """
        lab = cv.cvtColor(face_img, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)

        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_mejorado = clahe.apply(l)

        lab_mejorado = cv.merge((l_mejorado, a, b))
        return cv.cvtColor(lab_mejorado, cv.COLOR_LAB2BGR)



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

        logger.debug(
            f"Rostro alineado: src_points={src_points.tolist()}, dst_points={dst_points.tolist()}"
        )

        return aligned

    def preprocess(self, face_img: np.ndarray):
        """Convierte la imagen a RGB, normaliza los valores de píxel y devuelve un tensor listo para el modelo."""
        face_rgb = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype(np.float32) / 127.5 - 1.0
        logger.debug(f"Imagen preprocesada: shape={face_normalized.shape}")
        return np.expand_dims(face_normalized, axis=0)


def normalize(embedding: np.ndarray) -> np.ndarray:
    """Normaliza un embedding dividiéndolo por su norma L2."""
    logger.debug(f"Normalizando embedding: original_norm={np.linalg.norm(embedding)}")
    return embedding / np.linalg.norm(embedding)
