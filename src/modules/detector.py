import numpy as np
from insightface.app import FaceAnalysis
from modules.face import Face
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("detector")


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
        logger.info("Detector de rostros inicializado con éxito.")

    def detect(self, frame: np.ndarray):
        """Detecta rostros en un fotograma y devuelve una lista de objetos `Face`."""

        faces_raw = self.app.get(frame)
        faces = []
        for f in faces_raw:
            landmarks = f.landmark_2d_106 if f.landmark_2d_106 is not None else f.kps
            faces.append(Face(f.bbox, landmarks, f.det_score))
        logger.info(f"Rostros detectados: {len(faces)}")
        return faces

    def get_main_face(self, faces: list[Face]):
        """Selecciona el rostro principal de la lista, definido como el de mayor área."""

        face = max(faces, key=lambda f: f.w * f.h)
        logger.info("Rostro principal seleccionado.")
        return face
