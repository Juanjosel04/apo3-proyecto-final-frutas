"""Segmentación clásica de frutas.

Dos métodos:

- ``otsu``: el color del fondo se estima con la mediana de los píxeles del borde
  (en CIELab), se calcula la distancia de color de cada píxel al fondo (ΔE de
  CIE76) y se umbraliza con Otsu. Funciona bien con fondo uniforme (las fotos
  propias), pero falla con fondos con textura.
- ``grabcut`` (por defecto): la máscara inicial es la intersección de Otsu sobre
  ΔE y Otsu sobre la saturación (la fruta suele ser más saturada que pisos,
  paredes y mesas). Esa máscara se usa como "probable fruta", el borde como
  "fondo seguro", y GrabCut la refina con modelos de color por mezclas de
  gaussianas y un corte de grafo. Se agregó porque el EDA mostró que muchas
  imágenes del dataset tienen fondos con textura.

En ambos casos la máscara se limpia con apertura y cierre morfológicos y se
rellenan los huecos dibujando los contornos externos.

Para ``mixed_quality`` se conservan todos los componentes conexos grandes y,
si dos frutas se tocan, se separan con transformada de distancia + watershed.

Referencias: N. Otsu (1979) para el umbral; C. Rother, V. Kolmogorov y
A. Blake (2004) para GrabCut, usando la implementación de OpenCV; tutorial de
OpenCV (https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html) y ejemplo
de scikit-image "Watershed segmentation" para el esquema de watershed sobre la
transformada de distancia.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

LADO_TRABAJO = 512  # se segmenta a esta resolución y la máscara se reescala
LADO_GRABCUT = 256  # GrabCut es más costoso; a 256 px tarda ~0,1 s por imagen


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


def _otsu(canal: np.ndarray) -> np.ndarray:
    _, m = cv2.threshold(canal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m


def _grabcut(peq: np.ndarray, inicial: np.ndarray, iteraciones: int = 5) -> np.ndarray:
    """Refina ``inicial`` con GrabCut. El borde de la imagen se marca como fondo seguro."""
    h, w = peq.shape[:2]
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[inicial > 0] = cv2.GC_PR_FGD
    g = max(2, int(0.03 * min(h, w)))
    gc[:g] = gc[-g:] = cv2.GC_BGD
    gc[:, :g] = gc[:, -g:] = cv2.GC_BGD
    if np.count_nonzero(gc == cv2.GC_PR_FGD) < 50:
        return inicial
    modelo_fondo, modelo_fruta = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(peq, gc, None, modelo_fondo, modelo_fruta, iteraciones, cv2.GC_INIT_WITH_MASK)
    return np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def mascara_binaria(img_bgr: np.ndarray, metodo: str = "grabcut") -> np.ndarray:
    """Máscara de todo lo que no es fondo (puede tener varias frutas).

    Parameters
    ----------
    img_bgr : np.ndarray
        Imagen BGR.
    metodo : {"grabcut", "otsu"}
        ``otsu`` usa solo el umbral sobre la distancia al fondo; ``grabcut``
        la refina como se explica en el encabezado del módulo.

    Returns
    -------
    np.ndarray
        Máscara uint8 (0/255) del mismo tamaño que la imagen original.
    """
    h, w = img_bgr.shape[:2]
    lado = LADO_TRABAJO if metodo == "otsu" else LADO_GRABCUT
    peq, _ = _reducir(img_bgr, lado)
    m = _otsu(mapa_distancia_fondo(peq))
    if metodo == "grabcut":
        sat = cv2.cvtColor(cv2.GaussianBlur(peq, (5, 5), 0), cv2.COLOR_BGR2HSV)[..., 1]
        inicial = cv2.bitwise_and(m, _otsu(sat))
        if np.count_nonzero(inicial) < 0.01 * inicial.size:
            inicial = m  # fruta poco saturada (p. ej. con moho blanco)
        m = _grabcut(peq, inicial)
    elif metodo != "otsu":
        raise ValueError(f"Método desconocido: {metodo}")
    k = max(3, int(0.015 * max(peq.shape[:2])) | 1)
    m = _rellenar(_limpiar(m, k))
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def mascara_fruta(img_bgr: np.ndarray, metodo: str = "grabcut") -> np.ndarray:
    """Máscara de la fruta principal: el componente conexo de mayor área."""
    m = mascara_binaria(img_bgr, metodo)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    mayor = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(etiquetas == mayor, 255, 0).astype(np.uint8)


def _separar_tocandose(mascara: np.ndarray, umbral_rel: float, r_min: float = 0.0) -> np.ndarray:
    """Separa frutas en contacto con watershed sobre la transformada de distancia.

    La transformada de distancia D(p) vale la distancia de cada píxel de fruta
    al fondo, así que el centro de cada fruta redonda es un máximo local. Esos
    máximos son los marcadores y se inunda -D dentro de la máscara: la frontera
    entre dos frutas queda en el "cuello" donde D es mínima. Se descartan los
    máximos menores a ``umbral_rel`` veces el máximo de su componente (ruido en
    el borde) o menores a ``r_min`` píxeles (franjas delgadas del fondo, que así
    quedan como una sola región). Devuelve una imagen de etiquetas (0 = fondo).
    """
    dist = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)
    n_comp, comp = cv2.connectedComponents(mascara)
    maximos = np.zeros(n_comp, np.float32)
    np.maximum.at(maximos, comp.ravel(), dist.ravel())

    separacion = max(5, int(0.035 * max(mascara.shape)))
    picos = peak_local_max(dist, min_distance=separacion, labels=comp, exclude_border=False)
    marcadores = np.zeros(mascara.shape, np.int32)
    k = 0
    for y, x in picos:
        if dist[y, x] >= max(umbral_rel * maximos[comp[y, x]], r_min):
            k += 1
            marcadores[y, x] = k
    # Componentes sin ningún pico válido conservan una etiqueta propia
    for c in range(1, n_comp):
        if not np.any(marcadores[comp == c]):
            y, x = np.unravel_index(np.argmax(np.where(comp == c, dist, -1)), dist.shape)
            k += 1
            marcadores[y, x] = k
    return watershed(-dist, marcadores, mask=mascara > 0).astype(np.int32)


def separar_frutas(
    img_bgr: np.ndarray,
    area_min_rel: float = 0.005,
    margen: float = 0.08,
    separar_contacto: bool = True,
    umbral_picos: float = 0.35,
    metodo: str = "otsu",
    lado_min: int = 24,
    compacidad_min: float = 0.35,
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
        Fracción del máximo de la transformada de distancia (por componente)
        que debe superar un máximo local para ser semilla de una fruta.
    metodo : {"otsu", "grabcut"}
        Método de ``mascara_binaria``. Por defecto ``otsu``: en estas escenas el
        fondo ocupa casi todo el borde y su color se estima bien, mientras que
        GrabCut tiende a perder las frutas pequeñas.
    lado_min : int
        Se descartan recortes cuyo lado mayor (en la imagen original) sea menor
        a este valor: son demasiado pequeños para clasificar la calidad.
    compacidad_min : float
        Mínimo de radio inscrito / radio equivalente. Descarta franjas y bordes
        del fondo que sobreviven al umbral.

    Returns
    -------
    list[FrutaRecortada]
        Frutas ordenadas de izquierda a derecha y de arriba a abajo.
    """
    h, w = img_bgr.shape[:2]
    peq, escala = _reducir(img_bgr)
    m = mascara_binaria(peq, metodo)

    r_min = 0.4 * lado_min * escala  # radio inscrito mínimo en la imagen de trabajo
    if separar_contacto:
        etiquetas = _separar_tocandose(m, umbral_picos, r_min=r_min)
    else:
        _, etiquetas = cv2.connectedComponents(m)

    area_min = area_min_rel * peq.shape[0] * peq.shape[1]
    frutas = []
    for e in np.unique(etiquetas):
        if e <= 0:
            continue
        comp = np.where(etiquetas == e, 255, 0).astype(np.uint8)
        area = np.count_nonzero(comp)
        if area < area_min:
            continue
        # Compacidad: radio inscrito frente al radio del círculo de igual área.
        # En un círculo vale 1, en un banano ~0,5 y en una franja del fondo < 0,3.
        r_inscrito = cv2.distanceTransform(comp, cv2.DIST_L2, 5).max()
        if r_inscrito < max(compacidad_min * np.sqrt(area / np.pi), r_min):
            continue
        comp = _rellenar(comp)
        comp = cv2.resize(comp, (w, h), interpolation=cv2.INTER_NEAREST)
        x, y, bw, bh = cv2.boundingRect(comp)
        if max(bw, bh) < lado_min:
            continue
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
