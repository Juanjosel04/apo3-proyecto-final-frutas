"""App de clasificación de calidad y tamaño de frutas.

Uso (desde la raíz del repo, con el entorno activado):
    streamlit run app/app_streamlit.py

Recibe una foto (archivo o cámara) con una sola fruta sobre un fondo simple y
muestra la fruta, la calidad con su confianza y el tamaño relativo.
"""

from __future__ import annotations

import io
import os
import sys
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

from estilos import COLOR_CALIDAD, CSS  # noqa: E402
from src.config import FRUTAS_PROYECTO  # noqa: E402
from src.inference.predecir import UMBRAL_CONFIANZA, Clasificador  # noqa: E402

LADO_MAX = 1024
NOMBRE_TAMANO = {"pequeno": "Pequeño", "mediano": "Mediano", "grande": "Grande"}
OPCION_AUTO = "Detectar"

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


def con_contorno(img_bgr: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    grosor = max(2, max(img_bgr.shape[:2]) // 220)
    cv2.drawContours(rgb, contornos, -1, (255, 255, 255), grosor + 3)
    cv2.drawContours(rgb, contornos, -1, (22, 163, 74), grosor)
    return rgb


def tarjeta(etiqueta: str, valor: str, detalle: str = "", color: str = "inherit") -> None:
    st.markdown(f"""<div class="tarjeta"><div class="etiqueta">{etiqueta}</div>
        <div class="valor" style="color:{color}">{valor}</div>
        <div class="detalle">{detalle}</div></div>""", unsafe_allow_html=True)


def barras(probabilidades: dict[str, float], titulo: str) -> str:
    filas = "".join(
        f'<div class="fila"><span>{c.capitalize()}</span><span>{p:.0%}</span></div>'
        f'<div class="barra"><span style="width:{p * 100:.1f}%;background:{COLOR_CALIDAD[c]}"></span></div>'
        for c, p in probabilidades.items())
    return f'<div class="tarjeta"><div class="etiqueta">{titulo}</div><div style="margin-top:10px">{filas}</div></div>'


# Barra lateral
with st.sidebar:
    st.markdown("### Configuración")
    eleccion = st.segmented_control("Fruta", [OPCION_AUTO, *FRUTAS_PROYECTO], default=OPCION_AUTO,
                                    format_func=lambda f: f.capitalize())
    modelo = st.segmented_control("Modelo de calidad", ["svm", "cnn"], default="svm",
                                  format_func=lambda m: {"svm": "SVM", "cnn": "CNN"}[m])
    st.markdown(
        '<p class="nota">La SVM usa características de color, textura y forma de la fruta segmentada; '
        'la CNN aprende directamente del recorte. Si la fruta no se indica, la detecta un Random Forest.</p>',
        unsafe_allow_html=True)
    st.divider()
    st.markdown(
        f'<p class="nota"><b>Cómo tomar la foto:</b> una sola fruta, centrada y completa, desde arriba, '
        f'sobre un fondo claro y liso, con luz difusa y sin flash.<br><br>'
        f'Si la confianza es menor a {UMBRAL_CONFIANZA:.0%} la app sugiere revisión humana.</p>',
        unsafe_allow_html=True)

st.markdown("""<div class="cabecera"><h1>Calidad y tamaño de frutas</h1>
<p>Manzana, banano y naranja · segmentación clásica + SVM o CNN entrenada desde cero</p></div>""",
            unsafe_allow_html=True)

pestana_archivo, pestana_camara = st.tabs(["Subir foto", "Cámara"])
with pestana_archivo:
    archivo = st.file_uploader("Foto de la fruta", type=["jpg", "jpeg", "png", "webp"],
                               label_visibility="collapsed")
with pestana_camara:
    captura = st.camera_input("Toma una foto", label_visibility="collapsed")

entrada = captura or archivo
if entrada is None:
    st.markdown('<div class="tarjeta"><div class="etiqueta">Sin imagen</div>'
                '<div class="detalle">Sube una foto o usa la cámara para clasificar una fruta.</div></div>',
                unsafe_allow_html=True)
    st.stop()

clasificador = cargar_clasificador()
img = leer_imagen(entrada.getvalue())
fruta = None if eleccion in (None, OPCION_AUTO) else eleccion
try:
    with st.spinner("Segmentando y clasificando..."):
        p = clasificador.predecir(img, fruta=fruta, modelo=modelo or "svm")
except ValueError as error:
    st.error(f"No se pudo procesar la imagen: {error}")
    st.stop()

col_img, col_res = st.columns([1.15, 1], gap="large")
with col_img:
    with st.container(border=True):
        st.image(con_contorno(img, p.mascara), caption="Contorno de la fruta segmentada",
                 width="stretch")
        st.image(cv2.cvtColor(p.recorte, cv2.COLOR_BGR2RGB), caption="Recorte que recibe el modelo",
                 width=170)

with col_res:
    if p.revisar:
        motivo = ("la segmentación parece incompleta" if not p.segmentacion_ok
                  else f"la confianza es de {p.confianza:.0%}")
        st.markdown(f'<div class="aviso"><b>Revisión humana sugerida:</b> {motivo}.</div>',
                    unsafe_allow_html=True)
    tarjeta("Calidad", p.calidad.capitalize(),
            f"Confianza {p.confianza:.0%} · modelo {'SVM' if (modelo or 'svm') == 'svm' else 'CNN'}",
            COLOR_CALIDAD[p.calidad])
    st.markdown(barras(p.probabilidades, "Probabilidad por clase"), unsafe_allow_html=True)
    detalle_fruta = (f"Detectada automáticamente ({p.prob_fruta:.0%}); si no es correcta, indícala a la izquierda"
                     if p.fruta_automatica else "Indicada por el usuario")
    st.markdown(f"""<div class="tarjeta doble">
        <div><div class="etiqueta">Fruta</div><div class="valor chico">{p.fruta.capitalize()}</div>
        <div class="detalle">{detalle_fruta}</div></div>
        <div><div class="etiqueta">Tamaño relativo</div>
        <div class="valor chico">{NOMBRE_TAMANO.get(p.tamano, "—")}</div>
        <div class="detalle">Diámetro normalizado {p.diametro_norm:.3f}</div></div></div>""",
                unsafe_allow_html=True)
    st.markdown(f'<p class="nota">Procesado en {p.tiempo_ms:.0f} ms. El tamaño es relativo al encuadre: '
                f'para convertirlo a centímetros la foto debe tomarse a una distancia fija.</p>',
                unsafe_allow_html=True)

# Historial de la sesión
historial = st.session_state.setdefault("historial", [])
clave = (getattr(entrada, "file_id", None) or entrada.name, p.fruta, modelo)
if not historial or historial[-1]["clave"] != clave:
    historial.append({"clave": clave, "hora": datetime.now().strftime("%H:%M:%S"), "fruta": p.fruta,
                      "calidad": p.calidad, "confianza": round(p.confianza, 3),
                      "tamano": p.tamano, "modelo": modelo or "svm", "revisar": p.revisar})
with st.expander(f"Historial de la sesión ({len(historial)})"):
    tabla = pd.DataFrame(historial).drop(columns="clave")
    st.dataframe(tabla, hide_index=True, width="stretch")
    st.download_button("Descargar CSV", tabla.to_csv(index=False).encode("utf8"),
                       file_name="predicciones.csv", mime="text/csv")
