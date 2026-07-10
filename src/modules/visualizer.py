from modules.face import Face
import numpy as np
import cv2 as cv


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
