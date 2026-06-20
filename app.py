
# -*- coding: utf-8 -*-
# app.py — Analizador de Postura para Fitness en Gym
# Especializado en ejercicios y análisis de postura
# Estructura: 3 capas (Data Layer / Business Logic / Presentation Layer)
# Basado en: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python

import mediapipe as mp
import gradio as gr
import numpy as np
import os
from urllib.request import urlopen
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from mediapipe.tasks import python


# ─────────────────────────────────────────────────────────────────────────
# CAPA 1 — DATA LAYER
# El modelo se carga una sola vez cuando arranca la aplicación.
# ─────────────────────────────────────────────────────────────────────────

# Descargar modelo si no existe
modelo_path = "pose_landmarker_heavy.task"
if not os.path.exists(modelo_path):
    print("⏳ Descargando modelo de MediaPipe...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    with urlopen(url) as response:
        with open(modelo_path, 'wb') as out_file:
            out_file.write(response.read())
    print("✓ Modelo descargado.")

# Inicializar detector con ALTA CONFIANZA (0.9) para gym
# Justificación: en gym queremos CERTEZA, mejor fallar que cometer errores
base_options = python.BaseOptions(model_asset_path=modelo_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.9,  # 🏋️ ALTA CERTEZA para análisis de fitness
    min_pose_presence_confidence=0.9
)
detector_pose = vision.PoseLandmarker.create_from_options(options)
print("✓ Analizador de Postura listo (Confianza: 0.9)")


# ─────────────────────────────────────────────────────────────────────────
# CAPA 2 — BUSINESS LOGIC
# ─────────────────────────────────────────────────────────────────────────

def dibujar_esqueleto(imagen_rgb, detection_result):
    """Dibuja el esqueleto del cuerpo detectado."""
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(imagen_rgb)

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style
        )

    return annotated_image


def analizar_postura_fitness(imagen_entrada):
    """
    Análisis de postura especializado para ENTRENAMIENTO EN GYM.
    Devuelve: imagen anotada + diagnóstico detallado con interpretación.
    """
    try:
        # Convertir imagen numpy a formato MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_entrada)

        # Detectar pose landmarks
        detection_result = detector_pose.detect(mp_image)

        # Dibujar el esqueleto
        imagen_anotada = dibujar_esqueleto(imagen_entrada, detection_result)

        # Verificar si se detectó alguna pose
        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            mensaje = "[ADVERTENCIA] No se detectó figura humana. Asegúrate de:\n"
            mensaje += "- Estar de frente a la cámara\n"
            mensaje += "- Buena iluminación\n"
            mensaje += "- Postura completamente visible"
            return imagen_anotada, mensaje

        # Extraer landmarks del primer cuerpo detectado
        lista_landmarks = detection_result.pose_landmarks[0]

        # Puntos clave
        punto_hombro_derecho = lista_landmarks[12]
        punto_hombro_izquierdo = lista_landmarks[11]
        punto_cadera_derecha = lista_landmarks[24]
        punto_cadera_izquierda = lista_landmarks[23]
        punto_rodilla_derecha = lista_landmarks[26]
        punto_rodilla_izquierda = lista_landmarks[25]
        punto_cabeza = lista_landmarks[0]
        punto_tobillo_derecho = lista_landmarks[28]

        # ✅ MÉTRICAS DE CONSIGNA 1
        distancia_hombros = round(abs(punto_hombro_derecho.x - punto_hombro_izquierdo.x), 3)
        diferencia_rodillas = round(abs(punto_rodilla_derecha.y - punto_rodilla_izquierda.y), 3)
        altura_cuerpo = round(abs(punto_cabeza.y - punto_tobillo_derecho.y), 3)
        visibilidad = round(punto_hombro_derecho.visibility, 2)
        cadera_y = round(punto_cadera_derecha.y, 3)

        # ─────────────────────────────────────────────────────────
        # INTERPRETACIONES AUTOMÁTICAS
        # ─────────────────────────────────────────────────────────

        # Interpretación: Distancia hombros
        if distancia_hombros < 0.08:
            interp_hombros = "[ADVERTENCIA] Hombros muy juntos (encorvado)"
        elif 0.08 <= distancia_hombros <= 0.15:
            interp_hombros = "[OK] Normal"
        else:
            interp_hombros = "[INFO] Hombros muy separados"

        # Interpretación: Visibilidad
        if visibilidad >= 0.9:
            interp_visibilidad = "[OK] Excelente detección"
        elif visibilidad >= 0.7:
            interp_visibilidad = "[OK] Buena detección"
        elif visibilidad >= 0.5:
            interp_visibilidad = "[ADVERTENCIA] Parcialmente visible"
        else:
            interp_visibilidad = "[ERROR] Punto muy oscuro/tapado"

        # Interpretación: Desequilibrio postural
        if diferencia_rodillas < 0.03:
            interp_equilibrio = "[OK] Excelente - Rodillas perfectamente alineadas"
        elif diferencia_rodillas < 0.08:
            interp_equilibrio = "[OK] Bueno - Rodillas casi alineadas"
        elif diferencia_rodillas < 0.15:
            interp_equilibrio = "[ADVERTENCIA] Revisar - Peso desigual entre piernas"
        else:
            interp_equilibrio = "[ERROR] Desequilibrio evidente"

        # Interpretación: Altura del cuerpo
        if altura_cuerpo > 0.65:
            interp_altura = "[OK] De pie completamente erguido"
        elif altura_cuerpo > 0.50:
            interp_altura = "[ADVERTENCIA] Algo flexionado"
        elif altura_cuerpo > 0.35:
            interp_altura = "[POSICION] Muy agachado (sentadilla profunda)"
        else:
            interp_altura = "[POSICION] Casi en el suelo"

        # Diagnóstico general
        if diferencia_rodillas < 0.03 and altura_cuerpo > 0.65 and distancia_hombros >= 0.08:
            diagnostico = "[EXCELENTE] POSTURA EXCELENTE - Ideal para entrenar"
            color_diag = "[OK]"
        elif diferencia_rodillas < 0.08 and altura_cuerpo > 0.50:
            diagnostico = "[BUENO] POSTURA BUENA - Con pequeños ajustes"
            color_diag = "[OK]"
        elif diferencia_rodillas < 0.15 or altura_cuerpo < 0.50:
            diagnostico = "[REVISAR] REVISAR POSTURA - Ajusta alineación"
            color_diag = "[ADVERTENCIA]"
        else:
            diagnostico = "[CORRECCION] POSTURA REQUIERE CORRECCION"
            color_diag = "[ERROR]"

        # ─────────────────────────────────────────────────────────
        # CONSTRUCCIÓN DEL REPORTE FINAL
        # ─────────────────────────────────────────────────────────

        reporte = f"""
========================================================
{diagnostico}
========================================================

METRICAS DETECTADAS:
────────────────────────────────────────────────────────

1. DISTANCIA ENTRE HOMBROS
   Valor: {distancia_hombros}
   {interp_hombros}

2. VISIBILIDAD (CONFIANZA)
   Valor: {visibilidad} (0-1)
   {interp_visibilidad}

3. POSICION CADERA DERECHA
   Valor Y: {cadera_y}
   Rango normal: 0.4-0.6

4. DESEQUILIBRIO POSTURAL [MAS IMPORTANTE]
   Valor: {diferencia_rodillas}
   {interp_equilibrio}

5. ALTURA DEL CUERPO
   Valor: {altura_cuerpo}
   {interp_altura}

────────────────────────────────────────────────────────

CONSEJOS:
  - Mantén los hombros relajados hacia atrás
  - Rodillas alineadas verticalmente con cadera
  - Posición neutra de columna (no hiperextendida)
  - Distribuye el peso equitativamente entre ambas piernas
"""

        return imagen_anotada, reporte.strip()

    except Exception as e:
        return imagen_entrada.copy(), f"Error al procesar: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────
# CAPA 3 — PRESENTATION LAYER
# ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="🏋️ Analizador de Postura Fitness",
    theme=gr.themes.Soft(primary_hue="blue")
) as app_fitness:

    gr.Markdown("""
    # 🏋️ ANALIZADOR DE POSTURA PARA ENTRENAMIENTO
    **Análisis de postura en tiempo real con MediaPipe + IA**

    Optimizado para:
    - ✅ Ejercicios en gym
    - ✅ Fotos de buena calidad
    - ✅ Detección CERTEZA (0.9 confianza)
    """)

    with gr.Row():
        with gr.Column():
            entrada_imagen = gr.Image(
                label="📸 Foto de Postura",
                type="numpy",
                sources=["upload", "webcam"]
            )
            boton_analizar = gr.Button(
                "🔍 Analizar Postura",
                variant="primary",
                size="lg"
            )

        with gr.Column():
            salida_imagen = gr.Image(label="🦴 Esqueleto Detectado")
            salida_reporte = gr.Textbox(
                label="📋 Análisis Detallado",
                lines=12,
                interactive=False
            )

    gr.Markdown("""
    ---
    **⚙️ Configuración:**
    - Min Confidence: **0.9** (Alta certeza, mejor para gym)
    - Modelo: MediaPipe Pose Heavy
    - Puntos detectados: 33 landmarks
    """)

    # Conectar botón
    boton_analizar.click(
        fn=analizar_postura_fitness,
        inputs=entrada_imagen,
        outputs=[salida_imagen, salida_reporte]
    )


if __name__ == "__main__":
    app_fitness.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
