import cv2 as cv
import numpy as np
import json

from src.modules.config import P_KALMAN, Q_KALMAN, R_KALMAN


from src.modules.vision_core import (
    Detector,
    Tracker,
    Visualizer,
    Preprocessor,
    FaceNetEmbedder,
    normalize,
)

if __name__ == "__main__":
    detector = Detector()
    tracker = Tracker()
    viz = Visualizer()
    preprocessor = Preprocessor()
    embedder = FaceNetEmbedder("faceNet.onnx", preprocessor)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara")
        exit()

    embeddings = []
    frame_count = 0

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

                # Capturar embedding cada frame
                try:
                    embedding = embedder.get_embedding(frame, face)
                    embeddings.append(embedding)
                except ValueError as e:
                    # Landmarks insuficientes en este frame (p.ej. ángulo difícil): se salta
                    print(f"⚠️ Frame descartado: {e}")

                # Mostrar progreso
                cv.putText(
                    frame,
                    f"Frames capturados: {len(embeddings)}",
                    (30, 50),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Cuando llegues a 100 frames, salir
                if len(embeddings) >= 100:
                    break
            else:
                tracker.reset()

            cv.imshow("Registro Conductor", frame)
            frame_count += 1
            if cv.waitKey(1) & 0xFF in [27, ord("q")]:
                break

    finally:
        cap.release()
        cv.destroyAllWindows()

    # --- Procesamiento final ---
    if len(embeddings) >= 20:
        franjas = np.array_split(embeddings, 20)

        embeddings_promediados = []
        for franja in franjas:
            emb = np.mean(franja, axis=0)
            emb_norm = normalize(emb)
            embeddings_promediados.append(emb_norm)

        with open("conductor.json", "w") as f:
            json.dump([emb.tolist() for emb in embeddings_promediados], f)

        print(
            f"Se guardaron {len(embeddings_promediados)} embeddings normalizados en conductor.json"
        )
    elif embeddings:
        print(
            f"⚠️ Solo se capturaron {len(embeddings)} muestras (se necesitan al menos 20). "
            "No se guardó conductor.json — vuelve a ejecutar el registro."
        )
    else:
        print("⚠️ No se capturó ningún embedding. No se guardó conductor.json.")
