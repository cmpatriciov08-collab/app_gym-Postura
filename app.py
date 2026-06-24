
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
  --bg-deep: #0a0a0c;
  --bg-main: #111114;
  --bg-card: #18181d;
  --bg-card-hover: #1e1e24;
  --border-subtle: #26262e;
  --border-accent: #3a3a45;
  --text-primary: #f4f4f6;
  --text-secondary: #9ca0ad;
  --text-muted: #6b6f7a;
  --accent: #c8ff00;
  --accent-dim: rgba(200, 255, 0, 0.15);
  --accent-glow: rgba(200, 255, 0, 0.35);
  --danger: #ff5c4a;
  --danger-dim: rgba(255, 92, 74, 0.15);
  --radius-sm: 10px;
  --radius: 14px;
  --radius-lg: 20px;
  --radius-xl: 28px;
  --shadow-card: 0 1px 2px rgba(0,0,0,0.4), 0 8px 24px rgba(0,0,0,0.35);
  --shadow-elevated: 0 2px 4px rgba(0,0,0,0.5), 0 14px 36px rgba(0,0,0,0.45);
}

* {
  box-sizing: border-box;
}

html, body, .gradio-container {
  background: var(--bg-deep) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
  -webkit-font-smoothing: antialiased;
}

.gradio-container {
  max-width: 1280px !important;
  padding: 28px 32px 48px !important;
  margin: 0 auto !important;
}

/* ===== Header ===== */
.hero-block {
  position: relative;
  background: linear-gradient(160deg, rgba(255,255,255,0.03) 0%, transparent 55%),
              linear-gradient(135deg, #1a1b22 0%, #13141a 100%);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-xl);
  padding: 24px 32px;
  margin-bottom: 16px;
  overflow: hidden;
  box-shadow: var(--shadow-card);
}
.hero-block::before {
  content: "";
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse 60% 50% at 10% 0%, var(--accent-dim), transparent 70%);
  pointer-events: none;
  opacity: 0.7;
}

.hero-title {
  margin: 0;
  font-size: 1.85rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  line-height: 1.15;
  background: linear-gradient(180deg, #ffffff 0%, #d6d9e0 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-subtitle {
  margin: 10px 0 0;
  color: var(--text-secondary);
  font-size: 1.05rem;
  line-height: 1.5;
  max-width: 640px;
}
.hero-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 16px;
  padding: 7px 14px;
  background: var(--accent-dim);
  border: 1px solid rgba(200, 255, 0, 0.25);
  border-radius: 999px;
  color: var(--accent);
  font-size: 0.85rem;
  font-weight: 600;
  letter-spacing: 0.02em;
}

/* ===== Layout panels ===== */
.panel-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: 20px;
  box-shadow: var(--shadow-card);
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.panel-card:hover {
  border-color: var(--border-accent);
}

.panel-card .wrap,
.panel-card .block {
  border-radius: var(--radius) !important;
  background: transparent !important;
}

/* ===== Settings gear ===== */
.gear-hint {
  text-align: right;
  color: var(--text-muted);
  font-size: 0.9rem;
  font-weight: 500;
  margin: 0 0 8px;
  letter-spacing: 0.02em;
}

.settings-panel {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: 18px;
  box-shadow: var(--shadow-card);
}
.settings-panel .label {
  font-size: 0.8rem;
  color: var(--text-muted);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 12px;
}

/* ===== Inputs ===== */
.gr-button-primary, button.primary {
  background: linear-gradient(135deg, #c8ff00 0%, #a8e000 100%) !important;
  color: #0a0a0c !important;
  border: none !important;
  font-weight: 700 !important;
  letter-spacing: 0.01em !important;
  border-radius: var(--radius) !important;
  padding: 14px 28px !important;
  font-size: 1rem !important;
  box-shadow: 0 0 0 0 var(--accent-glow);
  transition: all 0.2s ease;
}
.gr-button-primary:hover, button.primary:hover {
  filter: brightness(1.08);
  box-shadow: 0 0 0 8px transparent, 0 4px 18px rgba(200, 255, 0, 0.25);
  transform: translateY(-1px);
}

.gr-image, .image-container {
  border-radius: var(--radius) !important;
  overflow: hidden;
  border: 1px solid var(--border-subtle) !important;
  background: var(--bg-card) !important;
}

/* ===== CTA ===== */
.cta-primary {
  display: flex;
  justify-content: center;
  margin-top: 10px;
}
.cta-primary button {
  width: 100%;
  max-width: 320px;
}

/* ===== Coach feedback ===== */
.coach-wrap {
  background: linear-gradient(180deg, rgba(255,255,255,0.025) 0%, transparent 40%),
              var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: 18px;
  box-shadow: var(--shadow-card);
}

.coach-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 18px;
  border-radius: 999px;
  font-size: 1rem;
  font-weight: 700;
  margin-bottom: 14px;
  letter-spacing: 0.01em;
  line-height: 1.3;
}
.coach-badge.ok {
  background: var(--accent-dim);
  color: var(--accent);
  border: 1px solid rgba(200, 255, 0, 0.3);
}
.coach-badge.warn {
  background: var(--danger-dim);
  color: #ff8a7a;
  border: 1px solid rgba(255, 92, 74, 0.3);
}

/* ===== Metric cards ===== */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 4px;
}

.metric-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.02) 0%, transparent 55%),
              #111119;
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius);
  padding: 14px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}
.metric-card:hover {
  border-color: var(--border-accent);
}

.metric-title {
  margin: 0;
  font-size: 0.78rem;
  color: var(--text-muted);
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.metric-value {
  margin: 8px 0 6px;
  font-size: 1.5rem;
  font-weight: 800;
  letter-spacing: -0.01em;
  color: var(--text-primary);
}
.metric-advice {
  margin: 0;
  font-size: 0.82rem;
  color: var(--text-secondary);
  line-height: 1.45;
}

.metric-score {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 10px;
}

.progress-track {
  flex: 1;
  height: 6px;
  border-radius: 999px;
  background: var(--border-subtle);
  overflow: hidden;
  box-shadow: inset 0 1px 2px rgba(0,0,0,0.35);
}
.progress-fill {
  height: 100%;
  border-radius: 999px;
  transition: width 0.6s cubic-bezier(0.22, 1, 0.36, 1);
}
.progress-fill.ok {
  background: linear-gradient(90deg, #a8e000, #c8ff00);
  box-shadow: 0 0 10px var(--accent-glow);
}
.progress-fill.warn {
  background: linear-gradient(90deg, #ff8a7a, #ff5c4a);
  box-shadow: 0 0 10px rgba(255, 92, 74, 0.25);
}

.score-label {
  font-size: 0.75rem;
  font-weight: 700;
  min-width: 32px;
  text-align: right;
  color: var(--text-secondary);
}

.metric-note {
  margin: 8px 0 0;
  font-size: 0.78rem;
  color: var(--text-muted);
  font-style: italic;
}

/* ===== Buttons global ===== */
.gr-button {
  border-radius: var(--radius) !important;
  font-weight: 600 !important;
  transition: all 0.2s ease;
}

.gr-button-secondary {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-accent) !important;
  color: var(--text-primary) !important;
}
.gr-button-secondary:hover {
  background: var(--bg-card-hover) !important;
  border-color: #555 !important;
}

/* ===== Accordion ===== */
.gr-accordion {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius) !important;
}
.gr-accordion .label {
  color: var(--text-secondary) !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
}

/* ===== Slider / Radio ===== */
.gr-slider, .gr-radio {
  color: var(--text-primary) !important;
}
input[type="range"] {
  accent-color: var(--accent) !important;
}

/* ===== Scrollbar ===== */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
::-webkit-scrollbar-track {
  background: transparent;
}
::-webkit-scrollbar-thumb {
  background: #2a2a35;
  border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover {
  background: #3a3a48;
}

@media (max-width: 860px) {
  .gradio-container {
    padding: 16px !important;
  }
  .hero-title {
    font-size: 1.5rem;
  }
  .metric-grid {
    grid-template-columns: 1fr;
  }
}

/* ===== Suggestions ===== */
.suggestions-wrap {
  background: linear-gradient(180deg, rgba(255,255,255,0.025) 0%, transparent 40%),
              var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: 18px;
  box-shadow: var(--shadow-card);
  margin-top: 16px;
}

.suggestions-title {
  margin: 0 0 12px;
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.suggestions-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.suggestion-item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 12px 14px;
  border-radius: var(--radius);
  font-size: 0.84rem;
  line-height: 1.5;
  border: 1px solid transparent;
}

.suggestion-item.critico {
  background: rgba(255, 92, 74, 0.08);
  border-color: rgba(255, 92, 74, 0.22);
  color: #ffcdc6;
}

.suggestion-item.recomendado {
  background: rgba(255, 170, 60, 0.08);
  border-color: rgba(255, 170, 60, 0.22);
  color: #ffd9a6;
}

.suggestion-item.ok {
  background: rgba(200, 255, 0, 0.06);
  border-color: rgba(200, 255, 0, 0.18);
  color: #d4eebb;
}

.suggestion-item.info {
  background: rgba(120, 160, 255, 0.08);
  border-color: rgba(120, 160, 255, 0.22);
  color: #c6d9ff;
}

.suggestion-icon {
  flex-shrink: 0;
  width: 20px;
  text-align: center;
  font-weight: 700;
}

.suggestion-text {
  flex: 1;
}

/* ===== Layout optimized ===== */
.settings-compact {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius) !important;
  padding: 14px !important;
  box-shadow: var(--shadow-card) !important;
}
.panel-label {
  color: var(--text-muted);
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin: 0 0 10px;
}

/* ===== Empty state ===== */
.empty-state {
  background: linear-gradient(180deg, rgba(255,255,255,0.025) 0%, transparent 40%), var(--bg-card);
  border: 1px dashed var(--border-accent);
  border-radius: var(--radius-lg);
  padding: 32px 24px;
  text-align: center;
  min-height: 320px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
}
.empty-icon {
  font-size: 2.6rem;
  line-height: 1;
  opacity: 0.85;
}
.empty-title {
  margin: 0;
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--text-primary);
}
.empty-desc {
  margin: 0;
  font-size: 0.9rem;
  color: var(--text-secondary);
  max-width: 280px;
}
.empty-tips {
  list-style: none;
  margin: 12px 0 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-size: 0.82rem;
  color: var(--text-muted);
}
.empty-tips li::before {
  content: "— ";
  color: var(--accent);
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
    <article class="metric-card">
      <p class="metric-title">{safe_title}</p>
      <p class="metric-value">{safe_value}</p>
      <div class="metric-score">
        <div class="progress-track">
          <div class="progress-fill {status}" style="width:{score}%"></div>
        </div>
        <span class="score-label">{score}%</span>
      </div>
      <p class="metric-advice">{safe_advice}</p>
    </article>
    """


def _generar_sugerencias_postura(distancia_hombros, desnivel_cadera, desequilibrio_rodillas, torso_inclinacion, alerta_plano, score_hombros, score_pelvis, score_rodillas, score_torso):
    sugerencias = []
    
    # === Alineacion de espalda ===
    if score_torso < 50:
        sugerencias.append(("critico", f"Corrección urgente: la inclinación del tronco ({torso_inclinacion:.1f}°) supera los límites seguros. Activa el core, retrae los escápulos y alinea el esternón con la pelvis antes de cada repetición."))
    elif score_torso < 70:
        sugerencias.append(("recomendado", f"Ajusta la alineación del tronco ({torso_inclinacion:.1f}°). Abre el pecho, retrae los escápulos y piensa en 'crecer' desde la coronilla para mantener la columna en posición neutral."))
    elif score_torso >= 85:
        sugerencias.append(("ok", "Alineación del tronco impecable: la columna se mantiene en posición neutral, reduciendo el riesgo de lesión y maximizando la transferencia de fuerza."))
    
    # === Balance de pelvis ===
    if score_pelvis < 50:
        sugerencias.append(("critico", f"Estabilización pélvica requerida (desnivel: {desnivel_cadera:.3f}). La pelvis presenta rotación. Coloca ambos pies al ancho de las caderas y nivela las crestas ilíacas antes de ejecutar el gesto."))
    elif score_pelvis < 70:
        sugerencias.append(("recomendado", f"Mejora el balance pélvico (desnivel: {desnivel_cadera:.3f}). Distribuye el peso equitativamente entre ambos apoyos y evita cargar más sobre un solo lado."))
    elif score_pelvis >= 85:
        sugerencias.append(("ok", "Balance pélvico sobresaliente: la pelvis estable actúa como el nexo firme entre el torso y las piernas."))
    
    # === Simetria de rodillas ===
    if score_rodillas < 50:
        sugerencias.append(("critico", f"Alineación de rodillas comprometida (desbalance: {desequilibrio_rodillas:.3f}). Asegura que ambas rodillas sigan la misma línea en flexión/extensión. Revisa si un pie está más adelantado o más abierto que el otro."))
    elif score_rodillas < 70:
        sugerencias.append(("recomendado", f"Controla la simetría de rodillas (diferencia: {desequilibrio_rodillas:.3f}). Mantén las rodillas alineadas con la punta de los pies y evita que una quede más alta o más baja que la otra."))
    elif score_rodillas >= 85:
        sugerencias.append(("ok", "Simetría de rodillas perfecta: el trabajo se distribuye equitativamente, protegiendo las articulaciones y maximizando la potencia del gesto."))
    
    # === Apertura de hombros ===
    if score_hombros < 50:
        sugerencias.append(("critico", f"Base de sustentación insuficiente (apertura: {distancia_hombros:.3f}). Separa los pies al ancho aproximado de los hombros para ganar estabilidad y evitar balanceos laterales."))
    elif score_hombros < 70:
        sugerencias.append(("recomendado", f"Ajusta la apertura de hombros (rango: {distancia_hombros:.3f}). Verifica que la separación te permita mantener el torso alineado sin compensaciones."))
    elif score_hombros >= 85:
        sugerencias.append(("ok", "Buena apertura de hombros: proporciona una base sólida y estable para la ejecución del gesto."))
    
    # === Recomendaciones combinadas ===
    if score_torso < 70 and score_pelvis < 70:
        sugerencias.append(("recomendado", "Tronco y pelvis desalineados: prioriza ejercicios de core como plancha isométrica, dead-bug o puente de glúteos para ganar estabilidad central."))
    
    if score_rodillas < 70 and score_pelvis < 70:
        sugerencias.append(("recomendado", "Descompensación en tren inferior: revisa la alineación pie-rodilla-cadera frente al espejo. Practica sentadillas sin peso para automatizar la postura."))
    
    if score_torso >= 85 and score_pelvis >= 85 and score_rodillas >= 85 and score_hombros >= 85:
        sugerencias.append(("ok", "Técnica sólida: postura lista para rendir. Puedes aumentar la intensidad con control y confianza."))
    
    if alerta_plano:
        sugerencias.insert(0, ("info", "La toma de cámara no es ideal para una evaluación precisa. Prioriza una foto lateral o frontal limpia, sin obstáculos y con buena iluminación."))
    
    return sugerencias


def _render_sugerencias_html(sugerencias):
    if not sugerencias:
        return ""
    
    iconos = {"critico": "!", "recomendado": "*", "ok": "+", "info": "i"}
    
    items_html = "".join([
        f"""<li class="suggestion-item {tipo}">
          <span class="suggestion-icon">{iconos.get(tipo, '-')}</span>
          <span class="suggestion-text">{html.escape(texto)}</span>
        </li>"""
        for tipo, texto in sugerencias
    ])
    
    return f"""<section class="suggestions-wrap">
      <h3 class="suggestions-title">Recomendaciones personalizadas</h3>
      <ul class="suggestions-list">{items_html}</ul>
    </section>"""


def _score_in_range(value, min_ok, max_ok):
    if min_ok <= value <= max_ok:
        return 100
    dist = min(abs(value - min_ok), abs(value - max_ok))
    return max(0, int(100 - dist * 1000))


def _validation_policy(confidence_threshold, validation_mode):
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


def _empty_state_html():
    return """
    <section class="empty-state">
      <div class="empty-icon">&#128247;</div>
      <p class="empty-title">Sin análisis aún</p>
      <p class="empty-desc">Subí una foto o usá la cámara para evaluar tu postura.</p>
      <ul class="empty-tips">
        <li>Foto de cuerpo completo, de frente o lateral.</li>
        <li>Buena iluminación y fondo limpio.</li>
        <li>Ropa cómoda que permita ver las articulaciones.</li>
      </ul>
    </section>
    """


def _prepare_image_for_mediapipe(imagen_entrada):
    if imagen_entrada is None:
        return None
    img = np.array(imagen_entrada)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def dibujar_esqueleto(imagen_rgb, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(imagen_rgb)

    point_style = drawing_utils.DrawingSpec(color=(245, 245, 245), thickness=1, circle_radius=1)
    line_style = drawing_utils.DrawingSpec(color=(200, 255, 0), thickness=2)

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
    imagen_prep = _prepare_image_for_mediapipe(imagen_entrada)
    if imagen_prep is None:
      raise ValueError("No se recibio imagen de entrada.")

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_prep)
    detection_result = detector.detect(mp_image)
    imagen_anotada = dibujar_esqueleto(imagen_prep, detection_result)

    if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
      banner = "Reencuadra tu postura para iniciar el escaneo"
      resumen = f"""
      <section class="coach-wrap">
        <span class="coach-badge warn">{html.escape(banner)}</span>
        <p class="metric-advice">Ubicate de cuerpo completo, con buena luz y fondo limpio.</p>
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
      banner = "Mejora la visibilidad antes de analizar"
      resumen = f"""
      <section class="coach-wrap">
        <span class="coach-badge warn">{html.escape(banner)}</span>
        <p class="metric-advice">La camara no ve bien suficientes articulaciones para una evaluacion fiable.</p>
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

    # Generar sugerencias personalizadas
    sugerencias = _generar_sugerencias_postura(
        distancia_hombros, desnivel_cadera, desequilibrio_rodillas, torso_inclinacion,
        alerta_plano, score_hombros, score_pelvis, score_rodillas, score_torso
    )
    sugerencias_html = _render_sugerencias_html(sugerencias)

    min_score = min(score_hombros, score_torso, score_pelvis, score_rodillas)
    if min_score >= 85 and not alerta_plano:
      badge_class = "ok"
      badge_text = "Tecnica solida: postura lista para rendir"
    else:
      badge_class = "warn"
      if score_torso < 70:
        badge_text = "Correccion sugerida: Alineacion de Espalda"
      elif score_pelvis < 70:
        badge_text = "Correccion sugerida: Estabiliza la pelvis"
      elif score_rodillas < 70:
        badge_text = "Correccion sugerida: Control de rodillas"
      else:
        badge_text = "Correccion sugerida: Ajuste tecnico general"

    plano_note = "<p class='metric-note'>Plano de camara no ideal: prioriza una toma lateral o frontal limpia.</p>" if alerta_plano else ""
    resumen = f"""
    <section class="coach-wrap">
      <span class="coach-badge {badge_class}">{html.escape(badge_text)}</span>
      <div class="metric-grid">{cards}</div>
      {sugerencias_html}
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
      "Error en el escaneo"
      "</span><p class='metric-advice'>Reintenta con otra imagen.</p></section>"
    )
    if imagen_entrada is not None:
        fallback = np.array(imagen_entrada)
        if fallback.ndim == 3 and fallback.shape[2] == 4:
            fallback = fallback[:, :, :3]
    else:
        fallback = np.zeros((480, 640, 3), dtype=np.uint8)
    return fallback, error_summary, f"Error al procesar: {str(e)}"


with gr.Blocks(title="Analizador de Postura Premium") as app_fitness:
    gr.HTML(
        """
        <section class='hero-block'>
          <h1 class='hero-title'>Analizador de Postura para Entrenamiento</h1>
          <p class='hero-subtitle'>Feedback visual estilo coach para mejorar tecnica, estabilidad y rendimiento.</p>
          <div class='hero-pill'>
            <span>&#9679;</span>
            <span>MediaPipe Pose &middot; Heavy Model</span>
          </div>
        </section>
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            gr.HTML("<p class='panel-label'>Configuración</p>")
            with gr.Column(elem_classes=["settings-compact"]):
                confidence_slider = gr.Slider(minimum=0.50, maximum=0.95, value=0.75, step=0.05, label="Confianza de detección")
                validation_mode = gr.Radio(choices=["flexible", "estricto"], value="flexible", label="Modo de validación")
            
            gr.Markdown("---") 
            
            entrada_imagen = gr.Image(label="Foto de postura", type="numpy", sources=["upload", "webcam"], height=360)
            with gr.Row():
                gr.ClearButton(value="Limpiar", components=[entrada_imagen])
                boton_analizar = gr.Button("Analizar postura", variant="primary", scale=3, size="lg")
            
            salida_imagen = gr.Image(label="Esqueleto detectado", height=360, interactive=False)

        with gr.Column(scale=8):
            salida_resumen = gr.HTML(value=_empty_state_html())
            
            with gr.Accordion("Datos avanzados", open=False):
                salida_avanzado = gr.Textbox(label="Metricas avanzadas", lines=8, interactive=False)

    boton_analizar.click(
        fn=analizar_postura_fitness,
        inputs=[entrada_imagen, confidence_slider, validation_mode],
        outputs=[salida_imagen, salida_resumen, salida_avanzado],
    )


if __name__ == "__main__":
  app_fitness.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CUSTOM_CSS)
