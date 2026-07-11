import cv2 as cv
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("visualizer")


class HudColors:
    """Paleta de colores del HUD, en formato BGR (el que usa OpenCV)."""

    AMARILLO = (0, 255, 255)
    VERDE = (0, 255, 0)
    ROJO = (0, 0, 255)
    CIAN = (255, 255, 0)


class Messages:
    """
    Clase para mostrar mensajes informativos en pantalla durante el proceso
    de registro y verificación del conductor.

    Encapsula la lógica de visualización con OpenCV (`cv.putText`) para
    mostrar el conteo de muestras, resultados de verificación y métricas
    de distancia/similitud.

    Parámetros
    ----------
    fuente : int, opcional
        Tipo de fuente de OpenCV (por defecto cv.FONT_HERSHEY_SIMPLEX).
    escala : float, opcional
        Escala del texto (por defecto 0.8).
    grosor : int, opcional
        Grosor del trazo del texto (por defecto 2).
    """

    def __init__(
        self, fuente=cv.FONT_HERSHEY_SIMPLEX, escala: float = 0.8, grosor: int = 2
    ):
        self.fuente = fuente
        self.escala = escala
        self.grosor = grosor

    def _dibujar(self, frame: np.ndarray, texto: str, posicion: tuple, color: tuple):
        """Método interno compartido: evita repetir los mismos 6 argumentos de cv.putText en cada método público."""
        cv.putText(frame, texto, posicion, self.fuente, self.escala, color, self.grosor)
        logger.debug(f"Texto dibujado: '{texto}' en {posicion} con color {color}")

    def mostrar_contador_muestras(
        self, frame: np.ndarray, muestras_actuales: int, total: int
    ):
        """Muestra en pantalla el número de muestras recolectadas sobre el total."""
        logger.debug(f"Mostrando contador de muestras: {muestras_actuales}/{total}")
        self._dibujar(
            frame,
            f"Muestras: {muestras_actuales}/{total}",
            (20, 40),
            HudColors.AMARILLO,
        )

    def mostrar_resultado_verificacion(self, frame: np.ndarray, autorizado: bool):
        """Muestra el resultado de la verificación en verde (autorizado) o rojo (no autorizado)."""
        texto = self.texto_resultado(autorizado)
        color = HudColors.VERDE if autorizado else HudColors.ROJO
        self._dibujar(frame, texto, (20, 80), color)
        logger.debug(f"Mostrando resultado de verificación: {texto}")

    def mostrar_distancia_promedio(self, frame: np.ndarray, distancia_promedio: float):
        """Muestra la distancia promedio calculada contra los embeddings de referencia, con tres decimales."""
        logger.debug(f"Mostrando distancia promedio: {distancia_promedio:.3f}")
        self._dibujar(
            frame, f"DIST: {distancia_promedio:.3f}", (20, 120), HudColors.CIAN
        )

    def texto_resultado(self, autorizado: bool) -> str:
        """Devuelve el texto del resultado de verificación según el booleano `autorizado`."""
        logger.debug(
            f"Generando texto de resultado: {'autorizado' if autorizado else 'no autorizado'}"
        )
        return "CONDUCTOR AUTORIZADO" if autorizado else "CONDUCTOR NO AUTORIZADO"
