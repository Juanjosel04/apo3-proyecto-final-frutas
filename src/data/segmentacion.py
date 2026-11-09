"""Segmentación clásica de frutas sobre fondo uniforme.

Idea: el fondo es casi uniforme, así que su color se estima con la mediana de
los píxeles del borde de la imagen (en CIELab). Luego se calcula, para cada
píxel, la distancia de color al fondo (ΔE de CIE76) y se umbraliza con Otsu.
La máscara se limpia con apertura y cierre morfológicos y se rellenan los
huecos dibujando los contornos externos.

Para ``mixed_quality`` se conservan todos los componentes conexos grandes y,
si dos frutas se tocan, se separan con transformada de distancia + watershed.

Referencias: N. Otsu (1979) para el umbral; documentación de OpenCV
(https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html) para el
esquema de watershed con marcadores.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

LADO_TRABAJO = 512  # se segmenta a esta resolución y la máscara se reescala


@dataclass
class FrutaRecortada:
    """Una fruta individual extraída de una imagen."""

    imagen: np.ndarray   # recorte BGR
    mascara: np.ndarray  # máscara del recorte (uint8, 0/255)
    caja: tuple[int, int, int, int]  # x, y, ancho, alto en la imagen original
    area: int  # píxeles de la fruta en la imagen original


def _reducir(img: np.ndarray, lado: int = LADO_TRABAJO) -> tuple[np.ndarray, float]:
    """Reduce la imagen para que su lado mayor sea ``lado``. Devuelve la escala."""
    h, w = img.shape[:2]
    escala = min(1.0, lado / max(h, w))
    if escala < 1.0:
        img = cv2.resize(img, (round(w * escala), round(h * escala)), interpolation=cv2.INTER_AREA)
    return img, escala


def color_fondo(img_lab: np.ndarray, grosor: float = 0.04) -> np.ndarray:
    """Mediana del color en una franja del borde (fracción ``grosor`` del lado)."""
    h, w = img_lab.shape[:2]
    g = max(2, int(grosor * min(h, w)))
    borde = np.concatenate([
        img_lab[:g].reshape(-1, 3),
        img_lab[-g:].reshape(-1, 3),
        img_lab[:, :g].reshape(-1, 3),
        img_lab[:, -g:].reshape(-1, 3),
    ])
    return np.median(borde, axis=0)


def mapa_distancia_fondo(img_bgr: np.ndarray) -> np.ndarray:
    """Distancia de color de cada píxel al fondo, escalada a uint8 (0-255).

    Se usa Lab porque la distancia euclidiana en ese espacio se aproxima a la
    diferencia de color percibida (ΔE CIE76). Se suaviza antes para reducir ruido.
    """
    suav = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    lab = cv2.cvtColor(suav, cv2.COLOR_BGR2LAB).astype(np.float32)
    fondo = color_fondo(lab)
    dist = np.linalg.norm(lab - fondo, axis=2)
    return cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _limpiar(mascara: np.ndarray, k: int) -> np.ndarray:
    """Apertura (quita ruido) y cierre (une huecos) con elemento elíptico k×k."""
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, elem)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, elem, iterations=2)


def _rellenar(mascara: np.ndarray) -> np.ndarray:
    """Rellena el interior de cada contorno externo (brillos, pedúnculo)."""
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lleno = np.zeros_like(mascara)
    cv2.drawContours(lleno, contornos, -1, 255, cv2.FILLED)
    return lleno


def mascara_binaria(img_bgr: np.ndarray) -> np.ndarray:
    """Máscara de todo lo que no es fondo (puede tener varias frutas).

    Returns
    -------
    np.ndarray
        Máscara uint8 (0/255) del mismo tamaño que la imagen original.
    """
    h, w = img_bgr.shape[:2]
    peq, _ = _reducir(img_bgr)
    dist = mapa_distancia_fondo(peq)
    _, m = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, int(0.015 * max(peq.shape[:2])) | 1)
    m = _rellenar(_limpiar(m, k))
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def mascara_fruta(img_bgr: np.ndarray) -> np.ndarray:
    """Máscara de la fruta principal: el componente conexo de mayor área."""
    m = mascara_binaria(img_bgr)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    mayor = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(etiquetas == mayor, 255, 0).astype(np.uint8)


def _separar_tocandose(mascara: np.ndarray, img_bgr: np.ndarray, umbral_rel: float) -> np.ndarray:
    """Separa frutas en contacto con watershed sobre la transformada de distancia.

    Los marcadores son los picos de la transformada de distancia (centros de
    cada fruta). Devuelve una imagen de etiquetas (0 = fondo).
    """
    dist = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)
    _, picos = cv2.threshold(dist, umbral_rel * dist.max(), 255, cv2.THRESH_BINARY)
    picos = picos.astype(np.uint8)
    n, marcadores = cv2.connectedComponents(picos)
    marcadores = marcadores + 1
    desconocido = cv2.subtract(mascara, picos)
    marcadores[desconocido == 255] = 0
    marcadores = cv2.watershed(img_bgr, marcadores.astype(np.int32))
    etiquetas = np.where(marcadores > 1, marcadores - 1, 0)
    return etiquetas.astype(np.int32)


def separar_frutas(
    img_bgr: np.ndarray,
    area_min_rel: float = 0.005,
    margen: float = 0.08,
    separar_contacto: bool = True,
    umbral_picos: float = 0.5,
) -> list[FrutaRecortada]:
    """Extrae cada fruta de una imagen con varias frutas (``mixed_quality``).

    Parameters
    ----------
    img_bgr : np.ndarray
        Imagen BGR original.
    area_min_rel : float
        Área mínima de un componente como fracción del área de la imagen.
    margen : float
        Margen alrededor de la caja envolvente, como fracción de su lado mayor.
    separar_contacto : bool
        Si es True, aplica watershed para separar frutas que se tocan.
    umbral_picos : float
        Fracción del máximo de la transformada de distancia usada como semilla.

    Returns
    -------
    list[FrutaRecortada]
        Frutas ordenadas de izquierda a derecha y de arriba a abajo.
    """
    h, w = img_bgr.shape[:2]
    peq, escala = _reducir(img_bgr)
    m = mascara_binaria(peq)

    if separar_contacto:
        etiquetas = _separar_tocandose(m, peq, umbral_picos)
    else:
        _, etiquetas = cv2.connectedComponents(m)

    area_min = area_min_rel * peq.shape[0] * peq.shape[1]
    frutas = []
    for e in np.unique(etiquetas):
        if e <= 0:
            continue
        comp = np.where(etiquetas == e, 255, 0).astype(np.uint8)
        if np.count_nonzero(comp) < area_min:
            continue
        comp = _rellenar(comp)
        comp = cv2.resize(comp, (w, h), interpolation=cv2.INTER_NEAREST)
        x, y, bw, bh = cv2.boundingRect(comp)
        pad = int(margen * max(bw, bh))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        frutas.append(FrutaRecortada(
            imagen=img_bgr[y0:y1, x0:x1].copy(),
            mascara=comp[y0:y1, x0:x1].copy(),
            caja=(x0, y0, x1 - x0, y1 - y0),
            area=int(np.count_nonzero(comp)),
        ))
    # Orden de lectura: por filas aproximadas y luego por x
    alto_fila = max(1, h // 6)
    frutas.sort(key=lambda f: (f.caja[1] // alto_fila, f.caja[0]))
    return frutas


def aplicar_mascara(img_bgr: np.ndarray, mascara: np.ndarray, color_fondo_salida=(0, 0, 0)) -> np.ndarray:
    """Deja solo la fruta y pinta el resto con ``color_fondo_salida``."""
    salida = np.full_like(img_bgr, color_fondo_salida, dtype=np.uint8)
    salida[mascara > 0] = img_bgr[mascara > 0]
    return salida


def recortar_fruta(img_bgr: np.ndarray, mascara: np.ndarray, margen: float = 0.05):
    """Recorta imagen y máscara a la caja envolvente de la fruta con un margen."""
    x, y, bw, bh = cv2.boundingRect(mascara)
    if bw == 0 or bh == 0:
        return img_bgr, mascara
    h, w = mascara.shape
    pad = int(margen * max(bw, bh))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    return img_bgr[y0:y1, x0:x1], mascara[y0:y1, x0:x1]
