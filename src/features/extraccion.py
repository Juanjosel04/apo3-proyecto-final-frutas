"""Características manuales de color, textura, forma y defectos de una fruta segmentada.

Todas se calculan solo sobre los píxeles de la máscara, para que el fondo no
influya. Entrada: recorte BGR y máscara de ``src.data.recortes``.

Grupos (el prefijo de cada columna indica el grupo):
- ``hist_``: histogramas normalizados de H (16 bins), S (8) y V (8).
- ``mom_``: media, desviación y asimetría de cada canal HSV y Lab.
- ``glcm_``: contraste, disimilitud, homogeneidad, energía y correlación de la
  matriz de co-ocurrencia de niveles de gris (Haralick et al., 1973), distancias 1 y 3.
- ``lbp_``: histograma de patrones binarios locales uniformes (Ojala et al., 2002).
- ``forma_``: área relativa, circularidad, excentricidad, solidez y relación de aspecto.
- ``def_``: proporción de píxeles oscuros, cafés y poco saturados (posibles defectos).

Uso (desde la raíz del repo, después de ``python -m src.data.recortes``):
    python -m src.features.extraccion
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from src.config import DATA_PROCESSED, RECORTES
from src.data.recortes import INDICE_RECORTES, leer_bgr

CARACTERISTICAS_KAGGLE = DATA_PROCESSED / "caracteristicas_kaggle.csv"

BINS_H, BINS_S, BINS_V = 16, 8, 8
NIVELES_GLCM = 32
LBP_P, LBP_R = 8, 1
UMBRAL_OSCURO = 80   # V < 80 (mismo criterio del EDA)


def _histogramas(hsv: np.ndarray, sel: np.ndarray) -> dict:
    h, s, v = hsv[..., 0][sel], hsv[..., 1][sel], hsv[..., 2][sel]
    out = {}
    for nombre, canal, bins, maximo in (("h", h, BINS_H, 180), ("s", s, BINS_S, 256),
                                        ("v", v, BINS_V, 256)):
        hist, _ = np.histogram(canal, bins=bins, range=(0, maximo))
        hist = hist / max(1, canal.size)
        out.update({f"hist_{nombre}{i:02d}": float(x) for i, x in enumerate(hist)})
    return out


def _momentos(hsv: np.ndarray, lab: np.ndarray, sel: np.ndarray) -> dict:
    out = {}
    for espacio, img, canales in (("hsv", hsv, "hsv"), ("lab", lab, "lab")):
        for i, c in enumerate(canales):
            x = img[..., i][sel].astype(np.float64)
            out[f"mom_{espacio}_{c}_media"] = float(x.mean())
            out[f"mom_{espacio}_{c}_std"] = float(x.std())
            out[f"mom_{espacio}_{c}_asim"] = float(skew(x)) if x.std() > 0 else 0.0
    return out


def _glcm(gris: np.ndarray, sel: np.ndarray) -> dict:
    # Nivel 0 = fondo; la fruta se cuantiza en 1..NIVELES-1 y luego se descarta el nivel 0
    q = np.zeros_like(gris, dtype=np.uint8)
    q[sel] = 1 + (gris[sel].astype(np.int32) * (NIVELES_GLCM - 1) // 256).astype(np.uint8)
    angulos = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    P = graycomatrix(q, distances=[1, 3], angles=angulos, levels=NIVELES_GLCM)
    P = P[1:, 1:].astype(np.float64)
    P /= np.maximum(P.sum(axis=(0, 1), keepdims=True), 1)
    out = {}
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        valores = graycoprops(P, prop)  # (distancias, ángulos)
        for j, d in enumerate((1, 3)):
            out[f"glcm_{prop}_d{d}"] = float(np.nan_to_num(valores[j].mean()))
    return out


def _lbp(gris: np.ndarray, sel: np.ndarray) -> dict:
    lbp = local_binary_pattern(gris, LBP_P, LBP_R, method="uniform")
    # Solo píxeles interiores, para no contar el borde fruta-fondo como textura
    interior = cv2.erode(sel.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    hist, _ = np.histogram(lbp[interior], bins=LBP_P + 2, range=(0, LBP_P + 2))
    hist = hist / max(1, interior.sum())
    return {f"lbp_{i:02d}": float(x) for i, x in enumerate(hist)}


def _forma(mascara: np.ndarray) -> dict:
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contornos:
        return {k: 0.0 for k in ("forma_area_rel", "forma_circularidad", "forma_excentricidad",
                                 "forma_solidez", "forma_aspecto")}
    c = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimetro = cv2.arcLength(c, True)
    circularidad = 4 * np.pi * area / perimetro ** 2 if perimetro > 0 else 0.0
    hull = cv2.contourArea(cv2.convexHull(c))
    solidez = area / hull if hull > 0 else 0.0
    m = cv2.moments(c)
    # Excentricidad de la elipse con los mismos momentos centrales de segundo orden
    a, b, cc = m["mu20"], m["mu11"], m["mu02"]
    raiz = np.sqrt(max(0.0, (a - cc) ** 2 + 4 * b ** 2))
    l1, l2 = (a + cc + raiz) / 2, (a + cc - raiz) / 2
    excentricidad = np.sqrt(1 - l2 / l1) if l1 > 0 else 0.0
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    aspecto = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
    return {"forma_area_rel": float(np.count_nonzero(mascara) / mascara.size),
            "forma_circularidad": float(min(circularidad, 1.0)),
            "forma_excentricidad": float(excentricidad), "forma_solidez": float(solidez),
            "forma_aspecto": float(aspecto)}


def _defectos(hsv: np.ndarray, sel: np.ndarray) -> dict:
    h, s, v = (hsv[..., i][sel].astype(np.int32) for i in range(3))
    n = max(1, h.size)
    cafe = (h >= 5) & (h <= 25) & (s >= 60) & (v < 150)
    return {"def_oscuro": float(np.sum(v < UMBRAL_OSCURO) / n),
            "def_cafe": float(np.sum(cafe) / n),
            "def_poco_saturado": float(np.sum(s < 40) / n)}


def extraer_caracteristicas(img_bgr: np.ndarray, mascara: np.ndarray) -> dict:
    """Vector de características de una fruta segmentada.

    Parameters
    ----------
    img_bgr : np.ndarray
        Recorte BGR (el fondo puede tener cualquier valor).
    mascara : np.ndarray
        Máscara uint8 (0/255) del mismo tamaño.

    Returns
    -------
    dict
        Nombre de la característica -> valor (78 características).
    """
    sel = mascara > 0
    if not sel.any():
        raise ValueError("La máscara está vacía")
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    out = {}
    out.update(_histogramas(hsv, sel))
    out.update(_momentos(hsv, lab, sel))
    out.update(_glcm(gris, sel))
    out.update(_lbp(gris, sel))
    out.update(_forma(mascara))
    out.update(_defectos(hsv, sel))
    return out


def _desde_archivos(recorte: str, mascara: str, raiz: Path) -> dict:
    img = leer_bgr(raiz / recorte)
    m = cv2.imdecode(np.fromfile(str(raiz / mascara), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return extraer_caracteristicas(img, m)


def columnas_por_grupo(columnas) -> dict[str, list[str]]:
    """Agrupa los nombres de columnas por prefijo (hist, mom, glcm, lbp, forma, def)."""
    grupos: dict[str, list[str]] = {}
    for c in columnas:
        pref = c.split("_")[0]
        if pref in {"hist", "mom", "glcm", "lbp", "forma", "def"}:
            grupos.setdefault(pref, []).append(c)
    return grupos


def main() -> None:
    idx = pd.read_csv(INDICE_RECORTES)
    idx = idx[idx["seg_ok"]].reset_index(drop=True)
    feats = Parallel(n_jobs=-2, batch_size=32)(
        delayed(_desde_archivos)(r, m, RECORTES) for r, m in zip(idx["recorte"], idx["mascara"]))
    X = pd.DataFrame(feats).round(6)
    df = pd.concat([idx[["ruta", "fruta", "calidad_dataset", "particion"]], X], axis=1)
    df.to_csv(CARACTERISTICAS_KAGGLE, index=False)
    print(f"{len(df)} filas x {X.shape[1]} características -> {CARACTERISTICAS_KAGGLE}")


if __name__ == "__main__":
    main()
