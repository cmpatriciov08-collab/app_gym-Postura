---
title: Analizador de Postura para Entrenamiento
emoji: 🏋️
colorFrom: gray
colorTo: indigo
sdk: gradio
python_version: "3.11"
app_file: app.py
pinned: false
---

# 🏋️ Analizador de Postura para Entrenamiento

Aplicación Gradio que utiliza **MediaPipe Pose Heavy** para detectar, analizar y evaluar la postura corporal en tiempo real a partir de una imagen o foto. Diseñada como proyecto de aplicación de visión artificial.

## Características

- **Detección de pose**: identifica 33 landmarks corporales con el modelo pesado de MediaPipe para mayor precisión.
- **Análisis biomecánico**: calcula ángulos de torso, simetría de pelvis, rodillas y apertura de hombros.
- **Feedback estilo coach**: tarjetas de métricas con puntuación y consejos técnicos.
- **Modos de validación**: Flexible (menos falsos rechazos) y Estricto (control técnico mayor).
- **Interfaz oscura premium**: tema visual adaptado para foco en la imagen y legibilidad.
- **Despliegue en HuggingFace**: lista para correr en Spaces sin instalación local.

## Stack

| Componente | Uso |
|------------|-----|
| **MediaPipe Tasks** | Inferencia de pose (modelo `.task` trackeado con Git LFS) |
| **Gradio 4+** | Interfaz web, carga de imagen/webcam, sliders y radio buttons |
| **NumPy + OpenCV** | Preprocesamiento de imagen y dibujo de esqueleto |
| **HuggingFace Spaces** | Hosting del servidor Gradio |

## Estructura del repositorio

```
008/002 - PRA/008 - vision_artificial_aplicada/
├── mi-app-gym/
│   ├── app.py                   # Lógica de detección, métricas y UI Gradio
│   ├── pose_landmarker_heavy.task  # Modelo MediaPipe (Git LFS)
│   ├── requirements.txt         # Dependencias Python
│   └── packages.txt             # Dependencias de sistema para HF Space
├── 04_Proyecto_Pose_y_Despliegue.ipynb  # Notebook del proyecto
└── ...
```

> **Nota**: La carpeta `mi-app-gym/` es la raíz del proyecto desplegado. El resto de la ruta forma parte de la estructura del curso.

## Uso local

```bash
cd "008/002 - PRA/008 - vision_artificial_aplicada/mi-app-gym"

# 1. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la app
python app.py
```

Luego abrir `http://localhost:7860`.

## Despliegue

La app está publicada en **HuggingFace Spaces**:

> https://huggingface.co/spaces/manuelcpv92/mi-app-gym/

Para volver a desplegar:

```bash
cd "mi-app-gym"
git add app.py
git commit -m "Actualizar app"
git push
```

## Flujo de análisis

1. **Entrada**: subir una foto o capturar desde webcam.
2. **Detección**: MediaPipe identifica la pose y calcula la visibilidad de 8 articulaciones clave.
3. **Validación**: aplica política de confianza y modo (Flexible/Estricto).
4. **Métricas**:
   - Alineación de espalda (ángulo torso)
   - Balance de pelvis (desnivel vertical)
   - Simetría de rodillas (diferencia de altura)
   - Apertura de hombros (distancia normalizada)
5. **Salida**: imagen anotada con esqueleto, badge de estado y tarjetas con puntuación.

## Créditos

Proyecto desarrollado por **Manuel Velásquez** como parte de la asignatura *Visión Artificial Aplicada* (PDI).