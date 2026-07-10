import numpy as np
from filterpy.kalman import KalmanFilter
from modules.face import Face
from configuraciones.config import P_KALMAN, Q_KALMAN, R_KALMAN


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

    def init_filter(
        self, face: Face, P: float = P_KALMAN, R: float = R_KALMAN, Q: float = Q_KALMAN
    ):
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
            self.init_filter(face, P, R, Q)
            return face

        medida = np.array([face.x, face.y, face.w, face.h], dtype=np.float32)

        self.kf.predict()
        self.kf.update(medida)

        # Extraemos solo los 4 primeros elementos (x, y, w, h)
        x, y, w, h = self.kf.x[:4].astype(int)

        # Actualizamos el objeto Face original con la predicción filtrada
        face.bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
        return face

    def reset(self):
        """Limpia el filtro para forzar una reinicialización en la próxima detección."""
        self.kf = None
