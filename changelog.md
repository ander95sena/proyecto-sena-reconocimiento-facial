
# V CHANGE LOG

Todos los cambios importantes del proyecto serán documentados en este archivo.

---

## Changelog v 2.5
 
Todos los cambios notables de este proyecto se documentan en este archivo.
El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).
 
 
#### Pendiente
- Anti-spoofing / detección de vida (liveness) para evitar ataques con foto o video.
- Cifrado de `conductor.json` (los embeddings son datos biométricos).
- Reemplazar `cv.getAffineTransform` (3 puntos) por `cv.estimateAffinePartial2D`
  (4+ puntos) para una alineación facial más robusta ante ruido de landmarks.
- Confirmar índice del landmark de la nariz en el modelo de 106 puntos para
  usarlo como cuarto punto de referencia en la alineación.
###  Refactor de módulos compartidos
 
#### Adicion 
- Nuevo módulo `vision_core.py` con las clases comunes del pipeline de visión
  (`Face`, `Detector`, `Tracker`, `Visualizer`, `Preprocessor`, `FaceNetEmbedder`,
  `normalize`), compartido entre `main.py` y `registro.py` para evitar
  duplicación de código y que los fixes queden desactualizados en un archivo
  y no en el otro.
- Validación de cantidad mínima de embeddings (20) antes de promediar y
  guardar `conductor.json` en `registro.py`.
#### Cambio
- `registro.py` ahora importa las clases desde `vision_core.py` en vez de
  definirlas localmente.
- `registro.py` usa `detector.get_main_face()` (criterio de mayor área) en
  vez de tomar `faces[0]` directamente, para ser consistente con `main.py`
  cuando hay más de una persona en cuadro.
- La función `normalize()` se movió de método de `FaceNetEmbedder` a función
  de módulo en `vision_core.py`, para tener una sola implementación
  compartida entre registro y verificación.
#### arreglos
- **Bug crítico:** si el registro se interrumpía antes de capturar 100
  frames, `np.array_split` generaba franjas vacías, `np.mean` de una franja
  vacía producía `NaN`, y el `conductor.json` se guardaba silenciosamente
  corrupto — sin ningún error visible, pero causando que un conductor
  legítimo nunca fuera autorizado más adelante.
- Landmarks insuficientes en `align_face` (fallback a 5 puntos en vez de
  106) ahora lanzan un `ValueError` explícito y controlado en vez de un
  `IndexError` no manejado a mitad de la captura.
- `tracker.reset()` agregado también en `registro.py` cuando no se detecta
  rostro, para no arrastrar el estado del Kalman entre personas distintas.
###  Corrección de bugs críticos en la verificación en vivo
 
#### arreglos
- **Bug de variable global `autorizado`:** `Messages.MensajeResultado()`
  dependía de una variable global que nunca se definía antes de su primer
  uso, causando `NameError` en la primera verificación. Corregido pasando
  `autorizado` como parámetro explícito.
- **Doble normalización de embeddings:** el embedding se normalizaba antes
  de guardarse en el `EmbeddingCollector` y de nuevo dentro de
  `FaceRecognition.verify()`. Corregido para normalizar una sola vez, al
  final, sobre el promedio.
- **Cálculo redundante de embedding por frame:** `embedder.get_embedding()`
  se ejecutaba en cada frame con rostro detectado, sin respetar el
  `skip_frames` configurado (la inferencia de FaceNet es la parte más cara
  del pipeline). Corregido con `EmbeddingCollector.debe_muestrear()`, que
  decide *antes* de calcular si corresponde tomar la muestra.
- **Parámetros `Q`/`R` invertidos en el Kalman:** `Tracker.update()` llamaba
  a `init_filter()` pasando los argumentos por posición en un orden distinto
  al de su firma, intercambiando el ruido del proceso (`Q`) con el ruido de
  la medición (`R`). Corregido pasando los argumentos por nombre.
- **`resultado` no se limpiaba al perder al conductor de cuadro:** si
  `faces` quedaba vacío, no había ningún `else` que reiniciara `resultado`,
  `distancia_promedio` ni el `Tracker`. Agregado contador
  `frames_sin_rostro` con reset completo tras `FRAMES_SIN_ROSTRO_PARA_RESET`
  frames consecutivos sin detección.
 
### Añadidos
- Pipeline de verificación facial en vivo (`main.py`): detección con
  InsightFace, tracking con Filtro de Kalman, alineación y embeddings con
  FaceNet (ONNX), verificación por distancia euclidiana contra un registro
  de embeddings, y envío de señal a un Arduino vía puerto serial.
- Script de registro de conductores (`registro.py`): captura 100 frames,
  los agrupa en 20 franjas de 5, promedia y normaliza cada franja, y guarda
  el resultado en `conductor.json`.


### [v2.0.0] - 2026-07-05

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
- Clase `serialArduinoDummy` para simulacion comunicación serial con el microcontrolador
- Sistema de documentación mediante docstrings.
- Configuración inicial de MkDocs para la documentación automática.
- Conexion serial real y simulada

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
