
# V CHANGE LOG

Todos los cambios importantes del proyecto serán documentados en este archivo.

---

## [v2.0.0] - 2026-07-05

### ✨ Agregado

- Arquitectura orientada a objetos para todo el sistema.
- Separación de responsabilidades mediante clases independientes.
- Clase `Detector` para la detección facial mediante InsightFace.
- Clase `Tracker` para el seguimiento temporal de rostros.
- Implementación del Filtro de Kalman utilizando una librería especializada.
- Clase `Face` para encapsular la información de cada rostro detectado.
- Clase `Preprocessor` para el alineamiento y preprocesamiento facial.
- Clase `FaceNetEmbedder` para la generación de embeddings faciales.
- Clase `EmbeddingCollector` para estabilizar los embeddings mediante múltiples muestras.
- Clase `FaceRecognition` para la comparación biométrica utilizando distancia euclidiana.
- Clase `Visualizer` para la visualización de información sobre la imagen.
- Clase `Messages` para centralizar los mensajes del sistema.
- Clase `serialArduino` para la comunicación serial con el microcontrolador.
- Sistema de documentación mediante docstrings.
- Configuración inicial de MkDocs para la documentación automática.

---

### 🔧 Mejorado

- Modularización completa del proyecto.
- Reducción del acoplamiento entre componentes.
- Organización del flujo principal del sistema.
- Mejor legibilidad del código.
- Mayor facilidad para realizar mantenimiento y futuras ampliaciones.
- Estructura preparada para incorporar nuevos módulos de procesamiento.
- Documentacion mas robusta

---

### 🛠 Corregido

- Ajustes en el seguimiento de rostros mediante Kalman.
- Correcciones en el flujo de reconocimiento.
- Correcciones menores en la comunicación serial.
- Mejor manejo de estados durante el procesamiento.
- Corrección de errores derivados del proceso de refactorización.

---

### ♻️ Refactorizado

- Migración desde una arquitectura basada en funciones hacia Programación Orientada a Objetos.
- Reorganización completa del código fuente.
- Separación de la lógica de negocio de la visualización.
- Eliminación de responsabilidades duplicadas.
- Mejora de nombres de clases, métodos y variables.
- Preparación de la arquitectura para futuras pruebas unitarias.

---

### 🗑 Eliminado

- Implementación manual del Filtro de Kalman.
- Código duplicado.
- Variables y funciones obsoletas.
- Dependencias innecesarias entre módulos.

---

### 📚 Documentación

- Documentación mediante docstrings en las clases principales.
- Inicio de la documentación automática utilizando MkDocs.
- Organización inicial de la documentación por módulos.
- Creación del archivo `CHANGELOG.md`.
- Actualización del `README.md`.

---

### 🚧 Trabajo en progreso

- Revision de logica de la normalizacion, parece  que esta duplicado
- Revision de propuestas o proceso de preprocesamiento de captura de frames, 
    iluminacion,escalas de grises.
- Evolucion a maquina de estados
- Revisar lo de los estados globales
- revsar lo de lo classdata de Face
- mejorar la conexion con el arduino

---

### 🚧 Proyecto futuros

Actualmente se encuentra en desarrollo la siguiente fase del proyecto:

- Face Quality Assessment.
- Normalización de iluminación.
- Evaluación de desenfoque.
- Detección de oclusiones.
- Head Pose Estimation.
- Liveness Detection.
- Confidence Manager.
- Máquina de estados.
- Sistema de Logging.
- Configuration Manager.
- Gestión de múltiples conductores.
- Optimización del rendimiento.
- Integración con CAN Bus.
- Migración hacia hardware automotriz.

---

## [v1.2.0] - 2026-06-21

### ✨ Agregado

- Implementación del Filtro de Kalman mediante librería especializada.

### 🔧 Cambiado

- Migración a Programación Orientada a Objetos.
- Separación inicial de responsabilidades.

### 🛠 Corregido

- Ajustes en la lógica general del sistema.
- Corrección de errores menores.

### 🗑 Eliminado

- Implementación manual del Filtro de Kalman.

### 📚 Documentación

- README con instrucciones básicas de instalación y uso.
- Documentación técnica inicial del proyecto.
