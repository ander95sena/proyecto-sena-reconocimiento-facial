import json
import logging
import cv2 as cv
import numpy as np

from configuraciones.config import (
    BAUDIOS,
    FRAMES_SIN_ROSTRO_PARA_RESET,
    MAX_EMBEDDINGS,
    P_KALMAN,
    PUERTOARDUINO,
    Q_KALMAN,
    R_KALMAN,
    RUTA_JSON,
    RUTA_MODELO,
    SKIP_FRAMES,
    UMBRAL_SIMILITUD,
)
from modules.conexionArduino import crear_conexion_arduino
from modules.detector import Detector
from modules.embeder import EmbeddingCollector, FaceNetEmbedder
from modules.faceRecognition import FaceRecognition
from modules.messages import Messages
from modules.preprosessing import Preprocessor
from modules.tracker import Tracker
from modules.visualizer import Visualizer

# Logs limpios en consola
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")


def procesar_deteccion_fatiga(frame, contador_frames):
    """Módulo liviano para el monitoreo de fatiga.
    
    Incluye un contador en tiempo real e indicador visual para confirmar
    su ejecución continua en pantalla.
    """
    # 1. Texto dinámico con el número de frame procesado en esta fase
    texto = f"SISTEMA DE FATIGA ACTIVO | Frames procesados: {contador_frames}"
    cv.putText(
        frame,
        texto,
        (20, frame.shape[0] - 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    # 2. Indicador visual de 'vida' (Punto verde parpadeante cada ~0.5 segundos)
    if (contador_frames // 15) % 2 == 0:
        cv.circle(frame, (frame.shape[1] - 30, 30), 8, (0, 255, 0), -1)


if __name__ == "__main__":
    # Inicialización de hardware y módulos de visión
    arduino = crear_conexion_arduino(PUERTOARDUINO, BAUDIOS)

    detector = Detector()
    tracker = Tracker()
    viz = Visualizer()
    messager = Messages()
    preprocessor = Preprocessor()
    embedder = FaceNetEmbedder(RUTA_MODELO, preprocessor)

    # Carga de la base de datos de rostros conocidos
    with open(RUTA_JSON, "r") as f:
        embeddings_registro = np.array(json.load(f), dtype=np.float32)

    recognizer = FaceRecognition(embeddings_registro, umbral=UMBRAL_SIMILITUD)
    collector = EmbeddingCollector(
        max_embeddings=MAX_EMBEDDINGS, skip_frames=SKIP_FRAMES
    )

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Error: no se pudo abrir la cámara")
        exit()

    # --- ESTADOS PERSISTENTES ---
    conductor_autorizado = False  # Bandera principal de sesión
    resultado = ""
    distancia_promedio = 0.0

    # Contador exclusivo para la Fase 1 (limpieza de buffers de autenticación)
    frames_sin_rostro_autenticacion = 0

    # Contador exclusivo para verificar visualmente la Fase 2 (Fatiga)
    contador_frames_fatiga = 0

    logger.info("Sistema iniciado. Esperando conductor para autenticación...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # =========================================================
            # FASE 1: AUTENTICACIÓN (Solo ejecuta mientras NO esté autorizado)
            # =========================================================
            if not conductor_autorizado:
                # El detector pesado y el tracker se ejecutan ÚNICAMENTE aquí
                faces = detector.detect(frame)

                if faces:
                    frames_sin_rostro_autenticacion = 0

                    face = detector.get_main_face(faces)
                    face = tracker.update(face, P_KALMAN, Q_KALMAN, R_KALMAN)

                    # Renderizado de la etapa de verificación
                    viz.draw_bbox(frame, face)
                    viz.draw_landmarks(frame, face)
                    viz.draw_score(frame, face)
                    viz.draw_eyes(frame, face)

                    if collector.debe_muestrear():
                        embedding = embedder.get_embedding(frame, face)
                        collector.add(embedding)

                    messager.mostrar_contador_muestras(
                        frame, collector.count(), MAX_EMBEDDINGS
                    )

                    if collector.is_ready():
                        embedding_actual = collector.get_average()
                        autorizado, distancia_promedio = recognizer.verify(
                            embedding_actual
                        )

                        if autorizado:
                            conductor_autorizado = True
                            resultado = messager.texto_resultado(True)
                            logger.info(
                                "✅ CONDUCTOR AUTORIZADO. Enviando '1' a Arduino y desactivando tracker."
                            )

                            # Envío único de la señal serial a Arduino
                            if arduino and arduino.is_open:
                                arduino.write(bytes([1]))
                        else:
                            resultado = messager.texto_resultado(False)
                            logger.warning("❌ No autorizado. Reintentando...")

                        collector.reset()

                else:
                    # Sin rostro durante la fase de autenticación
                    frames_sin_rostro_autenticacion += 1
                    if (
                        frames_sin_rostro_autenticacion
                        >= FRAMES_SIN_ROSTRO_PARA_RESET
                    ):
                        tracker.reset()
                        collector.reset()
                        resultado = ""
                        distancia_promedio = 0.0
                        frames_sin_rostro_autenticacion = 0

                # Renderizar estado de las pruebas de autenticación
                if resultado:
                    messager.mostrar_resultado_verificacion(frame, False)
                    messager.mostrar_distancia_promedio(frame, distancia_promedio)

            # =========================================================
            # FASE 2: MONITOREO DE FATIGA (Solo cuando YA está AUTORIZADO)
            # =========================================================
            else:
                # Se incrementa el contador de frames de la sesión de monitoreo
                contador_frames_fatiga += 1

                # Llamada al módulo de fatiga (Totalmente independiente de tracker/detector)
                procesar_deteccion_fatiga(frame, contador_frames_fatiga)

                # Notificación visual persistente de sesión autorizada
                messager.mostrar_resultado_verificacion(frame, True)
                messager.mostrar_distancia_promedio(frame, distancia_promedio)

            # Mostrar frame procesado en la ventana
            cv.imshow("Verificacion Conductor", frame)

            tecla = cv.waitKey(1) & 0xFF
            if tecla == 27 or tecla == ord("q"):
                break

    finally:
        cap.release()
        cv.destroyAllWindows()
        if arduino and arduino.is_open:
            arduino.close()