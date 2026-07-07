

# 🚗 Sistema Inteligente de Autorización de Conductor mediante Reconocimiento Facial


## 📖 Descripción

Este proyecto implementa un sistema inteligente de autenticación de conductores utilizando **visión por computador** e **inteligencia artificial**.

El objetivo es verificar la identidad del conductor antes de permitir el encendido del vehículo mediante reconocimiento facial.

Actualmente el sistema integra:

- 📷 Detección facial con **InsightFace**
- 🎯 Seguimiento de rostros mediante **Filtro de Kalman**
- 😀 Alineación facial automática
- 🧠 Generación de embeddings con **FaceNet**
- 📏 Comparación biométrica mediante distancia euclidiana
- 💾 Estabilización mediante múltiples embeddings
- 🔌 Comunicación serial con **Arduino**
- 🏗 Arquitectura orientada a objetos



## 🎯 Objetivos

- Autenticar conductores mediante biometría facial.
- Reducir falsas aceptaciones y falsos rechazos.
- Diseñar una arquitectura modular y escalable.
- Integrar el sistema con hardware automotriz.
- Evolucionar hacia un Driver Monitoring System (DMS).



## 🏗 Arquitectura del Sistema


                  Cámara
                     │
                     ▼
            Captura de Video
                     │
                     ▼
          Detección (InsightFace)
                     │
                     ▼
           Seguimiento (Kalman)
                     │
                     ▼
          Alineamiento Facial
                     │
                     ▼
         Generación de Embeddings
                (FaceNet)
                     │
                     ▼
        Embedding Collector
                     │
                     ▼
      Comparación Biométrica
                     │
                     ▼
      Comunicación con Arduino
                     │
                     ▼
        Autorización del Vehículo


## 📂 Estructura del Proyecto


📦 Proyecto

├── main.py
├── Arduino.py
├── conductor.json
├── faceNet.onnx
├── docs/
├── README.md
├── CHANGELOG.md
└── requirements.txt
```

---

## 🧩 Componentes Principales

| Clase | Responsabilidad |
|--------|-----------------|
| Detector | Detección facial mediante InsightFace |
| Tracker | Seguimiento mediante Filtro de Kalman |
| Face | Modelo de datos del rostro |
| Preprocessor | Alineación y normalización facial |
| FaceNetEmbedder | Generación de embeddings |
| EmbeddingCollector | Promedio temporal de embeddings |
| FaceRecognition | Comparación biométrica |
| Visualizer | Visualización del sistema |
| Messages | Gestión de mensajes |
| serialArduino | Comunicación serial |

---

## ⚙ Tecnologías Utilizadas

- Python
- OpenCV
- InsightFace
- FaceNet (ONNX)
- ONNX Runtime
- NumPy
- FilterPy
- PySerial
- Arduino IDE

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/ander95sena/carroIA.git
```



### 2. Crear entorno virtual

Windows

```bash
python -m venv .venv
```

Activar

```bash
.venv\Scripts\activate
```

---

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar

```bash
python main.py
```

---

## 📚 Documentación

La documentación técnica se genera automáticamente mediante **MkDocs** y **mkdocstrings**.

Para iniciar el servidor de documentación:

```bash
mkdocs serve
```

Generar documentación HTML:

```bash
mkdocs build
```

---

## 📈 Estado del Proyecto

| Módulo | Estado |
|---------|--------|
| Detección Facial | ✅ |
| Tracking | ✅ |
| Face Alignment | ✅ |
| FaceNet | ✅ |
| Reconocimiento | ✅ |
| Comunicación Serial | ✅ |
| Face Quality | 🚧 |
| Liveness | ⏳ |
| Head Pose | ⏳ |
| DMS | ⏳ |

---

## 🤝 Contribuciones

Actualmente este proyecto se encuentra en desarrollo activo.

Las sugerencias y mejoras son bienvenidas mediante Issues o Pull Requests.

---

# 📄 Licencia

Este proyecto se distribuye bajo la licencia MIT.

---

# 👨‍💻 Autor

** GRUPO SENA **


Proyecto de investigación y desarrollo en visión por computador e inteligencia artificial aplicada a la autenticación de conductores.
