"""App de clasificación de calidad y tamaño de frutas.

Uso (desde la raíz del repo, con el entorno activado):
    streamlit run app/app_streamlit.py

Tres modos de entrada: subir una foto, tomar una foto con la cámara o
clasificar en vivo lo que se muestra a la cámara. En cada caso se muestra la
fruta, la calidad con su confianza y el tamaño relativo.
"""

from __future__ import annotations

import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from estilos import COLOR_CALIDAD, COLOR_CONTORNO, CSS  # noqa: E402
from src.config import FRUTAS_PROYECTO  # noqa: E402
from src.inference.predecir import UMBRAL_CONFIANZA, Clasificador, Prediccion  # noqa: E402

LADO_MAX = 1024
NOMBRE_TAMANO = {"pequeno": "Pequeño", "mediano": "Mediano", "grande": "Grande"}
OPCION_AUTO = "Detectar"
MODOS = ["Subir foto", "Tomar foto", "En vivo"]
NOMBRE_MODELO = {"svm": "SVM", "cnn": "CNN"}

st.set_page_config(page_title="Calidad de frutas", page_icon=":material/nutrition:", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Cargando modelos...")
def cargar_clasificador() -> Clasificador:
    return Clasificador(usar_cnn=True)


def leer_imagen(datos: bytes) -> np.ndarray:
    """Bytes -> BGR, con la orientación EXIF corregida y el lado mayor en 1024 px."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(datos))).convert("RGB")
    img.thumbnail((LADO_MAX, LADO_MAX))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def con_contorno(img_bgr: np.ndarray, mascara: np.ndarray, calidad: str) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    grosor = max(2, max(img_bgr.shape[:2]) // 220)
    cv2.drawContours(rgb, contornos, -1, (255, 255, 255), grosor + 3, cv2.LINE_AA)
    cv2.drawContours(rgb, contornos, -1, COLOR_CONTORNO[calidad], grosor, cv2.LINE_AA)
    return rgb


def html_resultado(p: Prediccion, modelo: str) -> str:
    """Tarjetas de calidad, probabilidades y detalles en el estilo de la app."""
    aviso = ""
    if p.revisar:
        motivo = ("la segmentación parece incompleta" if not p.segmentacion_ok
                  else f"la confianza es de {p.confianza:.0%}")
        aviso = f'<div class="aviso"><b>Revisión humana sugerida</b> · {motivo}</div>'
    barras = "".join(
        f'<div class="fila"><span>{c.capitalize()}</span><span>{v:.0%}</span></div>'
        f'<div class="barra"><span style="width:{v * 100:.1f}%;background:{COLOR_CALIDAD[c]}"></span></div>'
        for c, v in p.probabilidades.items())
    origen_fruta = (f"detectada ({p.prob_fruta:.0%})" if p.fruta_automatica else "indicada")
    filas = [
        ("Fruta", f"{p.fruta.capitalize()} <small>{origen_fruta}</small>"),
        ("Tamaño relativo", NOMBRE_TAMANO.get(p.tamano, "—")),
        ("Diámetro normalizado", f"{p.diametro_norm:.3f}"),
        ("Modelo", NOMBRE_MODELO[modelo]),
        ("Tiempo", f"{p.tiempo_ms:.0f} ms"),
    ]
    lista = "".join(f'<div class="item"><span>{a}</span><span>{b}</span></div>' for a, b in filas)
    return f"""{aviso}
<div class="vidrio heroe"><div class="etiqueta">Calidad</div>
  <div class="valor" style="color:{COLOR_CALIDAD[p.calidad]}">{p.calidad.capitalize()}</div>
  <div class="detalle">Confianza {p.confianza:.0%}</div></div>
<div class="vidrio"><div class="etiqueta">Probabilidad por clase</div><div class="barras">{barras}</div></div>
<div class="vidrio lista">{lista}</div>"""


def vacio(titulo: str, texto: str) -> str:
    return f'<div class="vidrio vacio"><div class="valor chico">{titulo}</div><div class="detalle">{texto}</div></div>'


# Barra lateral
with st.sidebar:
    st.markdown('<div class="titulo-lateral">Ajustes</div>', unsafe_allow_html=True)
    eleccion = st.segmented_control("Fruta", [OPCION_AUTO, *FRUTAS_PROYECTO], default=OPCION_AUTO,
                                    format_func=lambda f: f.capitalize())
    modelo = st.segmented_control("Modelo de calidad", ["svm", "cnn"], default="svm",
                                  format_func=NOMBRE_MODELO.get) or "svm"
    st.markdown(
        '<div class="vidrio lista lateral">'
        '<div class="item"><span>SVM</span><span>color, textura y forma</span></div>'
        '<div class="item"><span>CNN</span><span>aprende del recorte</span></div>'
        '<div class="item"><span>Detectar</span><span>Random Forest</span></div></div>'
        '<p class="nota"><b>Cómo tomar la foto:</b> una sola fruta, centrada y completa, desde arriba, '
        'sobre un fondo claro y liso, con luz difusa y sin flash.</p>'
        f'<p class="nota">Si la confianza es menor a {UMBRAL_CONFIANZA:.0%} la app sugiere revisión humana.</p>',
        unsafe_allow_html=True)

fruta = None if eleccion in (None, OPCION_AUTO) else eleccion

st.markdown("""<div class="cabecera"><h1>Calidad de frutas</h1>
<p>Manzana, banano y naranja · segmentación clásica con SVM o CNN entrenada desde cero</p></div>""",
            unsafe_allow_html=True)
modo = st.segmented_control("Modo", MODOS, default=MODOS[0], label_visibility="collapsed",
                            key="modo") or MODOS[0]

# Modo en vivo: el video se procesa cuadro a cuadro y las tarjetas se refrescan solas
if modo == "En vivo":
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    from en_vivo import ProcesadorEnVivo

    clasificador = cargar_clasificador()
    col_video, col_res = st.columns([1.25, 1], gap="large")
    with col_video:
        ctx = webrtc_streamer(
            key="en-vivo", mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: ProcesadorEnVivo(clasificador),
            media_stream_constraints={"video": {"width": {"ideal": 960}}, "audio": False},
            rtc_configuration={"iceServers": []},
            async_processing=True,
            translations={"start": "Iniciar cámara", "stop": "Detener", "select_device": "Elegir cámara",
                          "device_ask_permission": "Permite el acceso a la cámara",
                          "device_not_available": "No hay cámara disponible",
                          "device_access_denied": "El navegador bloqueó la cámara"},
        )
        st.markdown('<p class="nota">Muestra una fruta frente a la cámara, sobre un fondo claro. '
                    'El contorno y la calidad se actualizan solos.</p>', unsafe_allow_html=True)
    with col_res:
        hueco = st.empty()
    if ctx.video_processor:
        ctx.video_processor.fruta, ctx.video_processor.modelo = fruta, modelo
    if not ctx.state.playing:
        hueco.markdown(vacio("Cámara apagada", "Oprime «Iniciar cámara» para clasificar en tiempo real."),
                       unsafe_allow_html=True)
    while ctx.state.playing and ctx.video_processor:
        p, error = ctx.video_processor.estado()
        hueco.markdown(html_resultado(p, modelo) if p else vacio("Buscando fruta", error or "Procesando..."),
                       unsafe_allow_html=True)
        time.sleep(0.4)
    st.stop()

# Modos con una foto
if modo == "Tomar foto":
    entrada = st.camera_input("Toma una foto", label_visibility="collapsed")
else:
    entrada = st.file_uploader("Foto de la fruta", type=["jpg", "jpeg", "png", "webp"],
                               label_visibility="collapsed")

if entrada is None:
    st.markdown(vacio("Sin imagen", "Sube una foto o usa la cámara para clasificar una fruta."),
                unsafe_allow_html=True)
    st.stop()

clasificador = cargar_clasificador()
img = leer_imagen(entrada.getvalue())
try:
    with st.spinner("Segmentando y clasificando..."):
        p = clasificador.predecir(img, fruta=fruta, modelo=modelo)
except ValueError as error:
    st.error(f"No se pudo procesar la imagen: {error}")
    st.stop()

col_img, col_res = st.columns([1.15, 1], gap="large")
with col_img:
    with st.container(border=True):
        st.image(con_contorno(img, p.mascara, p.calidad), caption="Contorno de la fruta segmentada",
                 width="stretch")
        st.image(cv2.cvtColor(p.recorte, cv2.COLOR_BGR2RGB), caption="Recorte que recibe el modelo",
                 width=150)
with col_res:
    st.markdown(html_resultado(p, modelo), unsafe_allow_html=True)
    st.markdown('<p class="nota">El tamaño es relativo al encuadre: para convertirlo a centímetros '
                'la foto debe tomarse a una distancia fija.</p>', unsafe_allow_html=True)

# Historial de la sesión
historial = st.session_state.setdefault("historial", [])
clave = (getattr(entrada, "file_id", None) or entrada.name, p.fruta, modelo)
if not historial or historial[-1]["clave"] != clave:
    historial.append({"clave": clave, "hora": datetime.now().strftime("%H:%M:%S"), "fruta": p.fruta,
                      "calidad": p.calidad, "confianza": round(p.confianza, 3),
                      "tamano": p.tamano, "modelo": modelo, "revisar": p.revisar})
with st.expander(f"Historial de la sesión ({len(historial)})"):
    tabla = pd.DataFrame(historial).drop(columns="clave")
    st.dataframe(tabla, hide_index=True, width="stretch")
    st.download_button("Descargar CSV", tabla.to_csv(index=False).encode("utf8"),
                       file_name="predicciones.csv", mime="text/csv")
