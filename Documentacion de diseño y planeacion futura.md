# Documento de Diseño del Sistema (SDD)
## Sistema Inteligente de Autorización de Conductor mediante Reconocimiento Facial

> **Versión:** 1.0 (Roadmap de desarrollo)

# Índice

1. Introducción
2. Objetivos
3. Alcance
4. Requisitos funcionales
5. Requisitos no funcionales
6. Arquitectura actual
7. Flujo del sistema
8. Descripción de módulos actuales
9. Evaluación del código
10. Mejoras propuestas
11. Roadmap de nuevos módulos
12. Arquitectura objetivo
13. UML propuesto
14. Fundamentos matemáticos
15. Integración automotriz
16. Optimización
17. Seguridad
18. Validación
19. Trabajo futuro
20. Bibliografía sugerida

---

# 1. Introducción

Este documento sirve como guía técnica para la evolución del sistema de reconocimiento facial desarrollado para autorizar el encendido de un vehículo. El objetivo es transformar el prototipo actual en una arquitectura robusta inspirada en sistemas Driver Monitoring System (DMS).

# 2. Objetivos

- Diseñar una arquitectura modular.
- Mejorar la robustez.
- Reducir falsas aceptaciones y rechazos.
- Facilitar la migración a hardware automotriz.

# 3. Arquitectura actual

```text
Cámara
 │
 ▼
Captura
 │
 ▼
InsightFace
 │
 ▼
Kalman
 │
 ▼
Face Alignment
 │
 ▼
FaceNet
 │
 ▼
Embedding Collector
 │
 ▼
Reconocimiento
 │
 ▼
Arduino
```

# 4. Clases actuales

## serialArduino
Responsabilidad:
- Comunicación serial.
- Inicialización.
- Envío de comandos.
- Cierre seguro.

Mejoras:
- Reconexión automática.
- CRC.
- ACK.
- Timeout.
- conexion dummy(realizado)
- clases con modularidad y facilidad para mas conexiones (completo)

## Face (realizado)
Contenedor de datos del rostro.
Mejora recomendada: convertir en @dataclass.

## Detector
Basado en InsightFace.
Responsabilidad:
- Detectar.
- Obtener landmarks.
- Seleccionar rostro principal.

Mejoras:
- Multi Face.
- Face Quality.

## Tracker
Filtro de Kalman.
Agregar:
- IDs persistentes.
- Reacquisition(realizado).

## Visualizer
Superposición gráfica
modularizar(en proceso).

## Preprocessor
- Alineación afín.
- RGB.
- Normalización.

## FaceNetEmbedder
Generación de embeddings.

## FaceRecognition
Comparación mediante distancia euclidiana.

## EmbeddingCollector
Promedio temporal para mejorar estabilidad.

## Messages
Mostrar resultados(en mejora)
Eliminar dependencias globales(completo)

# 5. Mejoras al código

## Prioridad alta

- Máquina de estados.
- Logging.
- Configuración externa(completado).
- Manejo de excepciones.
- Comunicación serial robusta(en mejora).
- Clase principal VehicleRecognitionSystem.
- Normalización única.
- Dataclass(realizado).
- Eliminar variables globales(realizado).

# 6. Roadmap de módulos

## Face Quality Assessment
Evalúa:
- Blur
- Iluminación
- Resolución
- Oclusión
- Área facial

## Head Pose Estimation
Calcula:
- Pitch
- Yaw
- Roll

## Liveness Detection
Protección contra:
- Fotos
- Videos
- Pantallas
- Máscaras

## Decision Fusion
Votación temporal de múltiples verificaciones.

## Confidence Manager
Fusiona:
- Distancia
- Pose
- Calidad
- Liveness

## Driver Database Manager
CRUD de conductores.

## Vehicle State Manager
Permite autenticación solo cuando el vehículo esté en estado seguro.

## Event Logger
Registro histórico.

## Metrics Manager
FPS, CPU, RAM, tiempos.

## Watchdog
Reinicio automático.

## Fail Safe
Modo seguro.

## Hardware Communication Manager
CRC, ACK, retransmisión.

## Multi Face Manager
Seguimiento de múltiples personas.

## Security Layer
Protección de datos y autenticación.

## CAN Interface
Migración futura hacia CAN Bus.

# 7. Arquitectura objetivo

```text
Camera
 │
Capture
 │
Face Quality
 │
Face Detector
 │
Multi Face
 │
Tracker
 │
Head Pose
 │
Liveness
 │
Alignment
 │
Embedding
 │
Decision Fusion
 │
Recognition
 │
Confidence
 │
Vehicle State
 │
State Machine
 │
Hardware Manager
 │
Arduino / ECU
```

# 8. UML propuesto

- Diagrama de clases
- Diagrama de secuencia
- Componentes
- Despliegue

# 9. Fundamentos matemáticos

Documentar:
- Filtro de Kalman
- Transformación afín
- Normalización L2
- Distancia euclidiana
- Embeddings
- EAR
- Anti-spoofing

# 10. Integración automotriz

Evolución propuesta:

Arduino
→ ESP32
→ STM32
→ ECU
→ CAN Bus
→ Driver Monitoring System

# 11. Seguridad

Implementar:
- Anti-spoofing
- Gestión de errores
- Protección de configuración
- Validación de integridad

# 12. Validación

Medir:
- Accuracy
- Precision
- Recall
- FAR
- FRR
- EER
- FPS
- Tiempo de inferencia

# 13. Trabajo futuro

Fase 1:
- Quality Assessment
- Pose
- Logging

Fase 2:
- Liveness
- Máquina de estados
- Confidence

Fase 3:
- CAN
- ECU
- Seguridad
- Hardware automotriz

# 14. Bibliografía sugerida

- FaceNet (Schroff et al.)
- ArcFace
- InsightFace
- ISO 26262
- ISO 21434
- SAE Driver Monitoring
- NHTSA Driver Monitoring
- ONNX Runtime Documentation
- OpenCV Documentation
