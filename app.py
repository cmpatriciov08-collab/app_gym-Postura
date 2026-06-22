
import html
import math
import os
from urllib.request import urlopen

import gradio as gr
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils


MODEL_PATH = "pose_landmarker_heavy.task"
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo de MediaPipe...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    with urlopen(url) as response:
        with open(MODEL_PATH, "wb") as out_file:
            out_file.write(response.read())
    print("Modelo descargado.")

DETECTOR_CACHE = {}


def get_detector(confidence_threshold=0.75):
  threshold = round(float(confidence_threshold), 2)
  if threshold in DETECTOR_CACHE:
    return DETECTOR_CACHE[threshold]

  base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
  options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    min_pose_detection_confidence=threshold,
    min_pose_presence_confidence=threshold,
  )
  detector = vision.PoseLandmarker.create_from_options(options)
  DETECTOR_CACHE[threshold] = detector
  return detector


detector_pose = get_detector(0.75)
print("Analizador de postura listo (confidence inicial: 0.75).")


CUSTOM_CSS = """
:root {
  --bg-main: #121214;
  --bg-card: #1A1B20;
  --bg-card-soft: #202228;
  --text-main: #F4F4F5;
  --text-soft: #A8ACB8;
  --accent-ok: #CCFF00;
  --accent-warn: #FF5733;
}

.gradio-container {
  background: radial-gradient(circle at 12% 20%, #23242A 0%, #121214 38%, #0F1013 100%);
  color: var(--text-main);
}

.hero-block {
  background: linear-gradient(135deg, #1C1D23 0%, #15161A 100%);
  border: 1px solid #2A2C34;
  border-radius: 16px;
  padding: 18px 20px;
  margin-bottom: 14px;
}

.hero-title {
  margin: 0;
  font-size: 1.75rem;
  font-weight: 800;
  letter-spacing: 0.01em;
}

.hero-subtitle {
  margin: 8px 0 0;
  color: var(--text-soft);
}

.panel-card {
  background: var(--bg-card);
  border: 1px solid #2A2C34;
  border-radius: 12px;
  padding: 14px;
}

.panel-card .wrap,
.panel-card .block {
  border-radius: 12px !important;
}

.cta-btn button {
  background: linear-gradient(90deg, #CCFF00 0%, #A3E635 100%) !important;
  color: #111214 !important;
  border: none !important;
  font-weight: 800 !important;
  letter-spacing: 0.01em !important;
}

.cta-btn button:hover {
  filter: brightness(1.06);
}

.coach-wrap {
  background: var(--bg-card-soft);
  border: 1px solid #2F323B;
  border-radius: 12px;
  padding: 12px;
}

.coach-badge {
  display: inline-block;
  padding: 8px 12px;
  border-radius: 999px;
  font-size: 0.88rem;
  font-weight: 700;
  margin-bottom: 12px;
}

.coach-badge.ok {
  background: rgba(204, 255, 0, 0.17);
  color: var(--accent-ok);
  border: 1px solid rgba(204, 255, 0, 0.45);
}

.coach-badge.warn {
  background: rgba(255, 87, 51, 0.14);
  color: #FF7A59;
  border: 1px solid rgba(255, 87, 51, 0.45);
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.metric-card {
  background: #17181D;
  border: 1px solid #2A2C34;
  border-radius: 12px;
  padding: 10px;
}

.metric-title {
  margin: 0;
  font-size: 0.85rem;
  color: var(--text-soft);
}

.metric-value {
  margin: 6px 0;
  font-size: 1rem;
  font-weight: 700;
}

.metric-advice {
  margin: 6px 0 0;
  font-size: 0.8rem;
  color: #D1D5E0;
}

.progress-track {
  width: 100%;
  height: 8px;
  border-radius: 999px;
  background: #2A2C34;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  border-radius: 999px;
}

.progress-fill.ok {
  background: linear-gradient(90deg, #A3E635 0%, #CCFF00 100%);
}

.progress-fill.warn {
  background: linear-gradient(90deg, #FF7A59 0%, #FF5733 100%);
}

.gear-hint {
  text-align: right;
  color: #C7CAD4;
  font-size: 0.88rem;
  margin: 0 0 6px;
}

@media (max-width: 900px) {
  .metric-grid {
    grid-template-columns: 1fr;
  }
}
"""


def _metric_status(score):
    return "ok" if score >= 70 else "warn"


def _build_metric_card(title, value, advice, score):
    safe_title = html.escape(title)
    safe_value = html.escape(value)
    safe_advice = html.escape(advice)
    score = max(0, min(100, int(score)))
    status = _metric_status(score)
    return f"""
    <article class=\"metric-card\">
      <p class=\"metric-title\">{safe_title}</p>
      <p class=\"metric-value\">{safe_value}</p>
      <div class=\"progress-track\">
        <div class=\"progress-fill {status}\" style=\"width:{score}%\"></div>
      </div>
      <p class=\"metric-advice\">{safe_advice}</p>
    </article>
    """


def _score_in_range(value, min_ok, max_ok):
    if min_ok <= value <= max_ok:
        return 100
    dist = min(abs(value - min_ok), abs(value - max_ok))
    return max(0, int(100 - dist * 1000))


def _validation_policy(confidence_threshold, validation_mode):
  # Umbral adaptativo: a mayor confidence, mayor exigencia de visibilidad.
  conf = float(confidence_threshold)
  vis_min = 0.35 + ((conf - 0.50) / 0.45) * 0.25
  vis_min = max(0.35, min(0.60, vis_min))

  mode = validation_mode.lower()
  if mode == "estricto":
    return {
      "vis_min": min(0.70, vis_min + 0.07),
      "coverage_min": 7,
      "trunk_min": min(0.70, vis_min + 0.02),
      "mode": "Estricto",
    }

  return {
    "vis_min": vis_min,
    "coverage_min": 6,
    "trunk_min": max(0.30, vis_min - 0.08),
    "mode": "Flexible",
  }


def dibujar_esqueleto(imagen_rgb, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(imagen_rgb)

    point_style = drawing_utils.DrawingSpec(color=(245, 245, 245), thickness=1, circle_radius=1)
    line_style = drawing_utils.DrawingSpec(color=(204, 255, 0), thickness=1)

    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=point_style,
            connection_drawing_spec=line_style,
        )

    return annotated_image


def analizar_postura_fitness(imagen_entrada, confidence_threshold, validation_mode):
  try:
    detector = get_detector(confidence_threshold)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_entrada)
    detection_result = detector.detect(mp_image)
    imagen_anotada = dibujar_esqueleto(imagen_entrada, detection_result)

    if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
      banner = "⚠️ Correccion sugerida: Reencuadra tu postura para iniciar el escaneo"
      resumen = f"""
      <section class=\"coach-wrap\">
        <span class=\"coach-badge warn\">{html.escape(banner)}</span>
        <p class=\"metric-advice\">Ubicate de cuerpo completo, con buena luz y fondo limpio.</p>
      </section>
      """
      return imagen_anotada, resumen, f"No se detectaron landmarks. confidence={confidence_threshold:.2f}"

    lms = detection_result.pose_landmarks[0]
    i = {
      "hombro_d": 12,
      "hombro_i": 11,
      "cadera_d": 24,
      "cadera_i": 23,
      "rodilla_d": 26,
      "rodilla_i": 25,
      "tobillo_d": 28,
      "tobillo_i": 27,
    }

    req = ["hombro_d", "hombro_i", "cadera_d", "cadera_i", "rodilla_d", "rodilla_i", "tobillo_d", "tobillo_i"]

    vis = {k: float(lms[i[k]].visibility) for k in req}
    min_vis = min(vis.values())
    policy = _validation_policy(confidence_threshold, validation_mode)
    valid_count = sum(1 for v in vis.values() if v >= policy["vis_min"])
    trunk_points = ["hombro_d", "hombro_i", "cadera_d", "cadera_i"]
    trunk_ok = all(vis[p] >= policy["trunk_min"] for p in trunk_points)

    if valid_count < policy["coverage_min"] or not trunk_ok:
      banner = "⚠️ Correccion sugerida: Mejora visibilidad antes de analizar"
      resumen = f"""
      <section class=\"coach-wrap\">
        <span class=\"coach-badge warn\">{html.escape(banner)}</span>
        <p class=\"metric-advice\">La camara no ve bien suficientes articulaciones para una evaluacion fiable.</p>
      </section>
      """
      return imagen_anotada, resumen, (
        f"Visibilidad minima detectada: {min_vis:.2f} | "
        f"coverage={valid_count}/8 (min {policy['coverage_min']}) | "
        f"trunk_ok={trunk_ok} (min {policy['trunk_min']:.2f}) | "
        f"vis_min={policy['vis_min']:.2f} | "
        f"confidence={confidence_threshold:.2f} | modo={policy['mode']}"
      )

    hombro_d = lms[i["hombro_d"]]
    hombro_i = lms[i["hombro_i"]]
    cadera_d = lms[i["cadera_d"]]
    cadera_i = lms[i["cadera_i"]]
    rodilla_d = lms[i["rodilla_d"]]
    rodilla_i = lms[i["rodilla_i"]]

    distancia_hombros = abs(hombro_d.x - hombro_i.x)
    desnivel_cadera = abs(cadera_d.y - cadera_i.y)
    desequilibrio_rodillas = abs(rodilla_d.y - rodilla_i.y)

    cadera_mid_x = (cadera_d.x + cadera_i.x) / 2
    cadera_mid_y = (cadera_d.y + cadera_i.y) / 2
    hombro_mid_x = (hombro_d.x + hombro_i.x) / 2
    hombro_mid_y = (hombro_d.y + hombro_i.y) / 2
    torso_inclinacion = math.degrees(
      math.atan2(abs(hombro_mid_x - cadera_mid_x), abs(cadera_mid_y - hombro_mid_y) + 1e-8)
    )

    vis_izq = (hombro_i.visibility + cadera_i.visibility + rodilla_i.visibility) / 3
    vis_der = (hombro_d.visibility + cadera_d.visibility + rodilla_d.visibility) / 3
    alerta_plano = abs(vis_izq - vis_der) > 0.35

    score_hombros = _score_in_range(distancia_hombros, 0.08, 0.15)
    score_torso = _score_in_range(torso_inclinacion, 0.0, 18.0)
    score_pelvis = _score_in_range(desnivel_cadera, 0.0, 0.03)
    score_rodillas = _score_in_range(desequilibrio_rodillas, 0.0, 0.03)

    cards = "".join([
      _build_metric_card(
        "Alineacion de espalda",
        f"{torso_inclinacion:.1f} grados",
        "Manten el tronco neutro y estable en todo el movimiento." if score_torso >= 70 else "Abre el pecho y retrae los hombros antes de iniciar el empuje.",
        score_torso,
      ),
      _build_metric_card(
        "Balance de pelvis",
        f"desnivel {desnivel_cadera:.3f}",
        "Pelvis estable: base solida para transferir fuerza." if score_pelvis >= 70 else "Nivela la pelvis y distribui la carga en ambos apoyos.",
        score_pelvis,
      ),
      _build_metric_card(
        "Simetria de rodillas",
        f"diferencia {desequilibrio_rodillas:.3f}",
        "Rodillas alineadas: excelente control del eje." if score_rodillas >= 70 else "Evita que una rodilla quede mas alta que la otra.",
        score_rodillas,
      ),
      _build_metric_card(
        "Apertura de hombros",
        f"rango {distancia_hombros:.3f}",
        "Buena apertura para un gesto potente." if score_hombros >= 70 else "Separa un poco mas los hombros para ganar estabilidad.",
        score_hombros,
      ),
    ])

    min_score = min(score_hombros, score_torso, score_pelvis, score_rodillas)
    if min_score >= 85 and not alerta_plano:
      badge_class = "ok"
      badge_text = "✅ Tecnica solida: postura lista para rendir"
    else:
      badge_class = "warn"
      if score_torso < 70:
        badge_text = "⚠️ Correccion sugerida: Alineacion de Espalda"
      elif score_pelvis < 70:
        badge_text = "⚠️ Correccion sugerida: Estabiliza la pelvis"
      elif score_rodillas < 70:
        badge_text = "⚠️ Correccion sugerida: Control de rodillas"
      else:
        badge_text = "⚠️ Correccion sugerida: Ajuste tecnico general"

    plano_note = "<p class='metric-advice'>Plano de camara no ideal: prioriza una toma lateral o frontal limpia.</p>" if alerta_plano else ""
    resumen = f"""
    <section class=\"coach-wrap\">
      <span class=\"coach-badge {badge_class}\">{html.escape(badge_text)}</span>
      <div class=\"metric-grid\">{cards}</div>
      {plano_note}
    </section>
    """

    avanzado = "\n".join([
      f"distancia_hombros={distancia_hombros:.3f}",
      f"desnivel_cadera={desnivel_cadera:.3f}",
      f"desequilibrio_rodillas={desequilibrio_rodillas:.3f}",
      f"torso_inclinacion_deg={torso_inclinacion:.1f}",
      f"visibilidad_minima={min_vis:.2f}",
      f"coverage={valid_count}/8",
      f"coverage_min={policy['coverage_min']}",
      f"trunk_ok={trunk_ok}",
      f"trunk_min={policy['trunk_min']:.2f}",
      f"vis_min_umbral={policy['vis_min']:.2f}",
      f"alerta_plano={alerta_plano}",
      f"confidence_umbral={confidence_threshold:.2f}",
      f"modo_validacion={policy['mode']}",
    ])

    return imagen_anotada, resumen, avanzado

  except Exception as e:
    error_summary = (
      "<section class='coach-wrap'><span class='coach-badge warn'>"
      "⚠️ Correccion sugerida: Error en el escaneo"
      "</span><p class='metric-advice'>Reintenta con otra imagen.</p></section>"
    )
    return imagen_entrada.copy(), error_summary, f"Error al procesar: {str(e)}"


with gr.Blocks(title="Analizador de Postura Premium") as app_fitness:
    gr.HTML(
        """
        <section class='hero-block'>
          <h1 class='hero-title'>Analizador de Postura para Entrenamiento</h1>
          <p class='hero-subtitle'>Feedback visual estilo coach para mejorar tecnica, estabilidad y rendimiento.</p>
        </section>
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            pass
        with gr.Column(scale=2):
            gr.HTML("<p class='gear-hint'>⚙ Ajustes</p>")
            with gr.Accordion("Panel de configuracion", open=False):
                confidence_slider = gr.Slider(
                    minimum=0.50,
                    maximum=0.95,
                    value=0.75,
                    step=0.05,
                    label="Confidence de deteccion",
                    info="Sube para mayor rigor. Baja para detectar mejor en fotos complejas.",
                )
                validation_mode = gr.Radio(
                    choices=["Flexible", "Estricto"],
                    value="Flexible",
                    label="Modo de validacion",
                    info="Flexible reduce falsos rechazos. Estricto aumenta control tecnico.",
                )
                gr.Markdown("Precisión de escaneo: regulable ⚙")
                gr.Markdown("Modo: Analisis estatico para entrenamiento en gym")
                gr.Markdown("Detalle tecnico disponible en Ver datos avanzados")

    with gr.Row():
        with gr.Column(elem_classes=["panel-card"]):
            entrada_imagen = gr.Image(
                label="Foto de postura",
                type="numpy",
                sources=["upload", "webcam"],
            )
            boton_analizar = gr.Button(
                "Analizar Postura",
                elem_classes=["cta-btn"],
                size="lg",
            )

        with gr.Column(elem_classes=["panel-card"]):
            salida_imagen = gr.Image(label="Esqueleto detectado")
            salida_resumen = gr.HTML(label="Feedback del coach")
            with gr.Accordion("Ver datos avanzados", open=False):
                salida_avanzado = gr.Textbox(label="Datos avanzados", lines=8, interactive=False)

    boton_analizar.click(
        fn=analizar_postura_fitness,
      inputs=[entrada_imagen, confidence_slider, validation_mode],
        outputs=[salida_imagen, salida_resumen, salida_avanzado],
    )


if __name__ == "__main__":
  app_fitness.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CUSTOM_CSS)
