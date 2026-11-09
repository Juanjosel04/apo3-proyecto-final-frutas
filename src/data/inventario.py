"""Inventario del dataset de Kaggle: una fila por imagen con fruta, calidad y medidas básicas.

El dataset (FruitNet, publicado en Kaggle por R. Park) organiza las imágenes en
carpetas por calidad (Good, Bad, Mixed) y por fruta. Los nombres exactos de las
carpetas cambian entre versiones, así que la fruta y la calidad se deducen
buscando palabras clave en la ruta.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import EXTENSIONES

FRUTAS_DATASET = {
    "apple": "manzana",
    "banana": "banano",
    "guava": "guayaba",
    "lime": "lima",
    "lemon": "lima",  # en mixed_quality la carpeta se llama Lemon
    "orange": "naranja",
    "pomegranate": "granada",
}
CALIDADES_DATASET = {"good": "good", "bad": "bad", "mixed": "mixed"}


def _buscar_clave(partes: list[str], claves: dict) -> str | None:
    """Devuelve el valor de la primera clave que aparezca en alguna parte de la ruta."""
    for parte in reversed(partes):
        p = parte.lower()
        for clave, valor in claves.items():
            if clave in p:
                return valor
    return None


def listar_imagenes(raiz: Path) -> list[Path]:
    """Todas las imágenes bajo ``raiz`` (recursivo)."""
    return sorted(p for p in Path(raiz).rglob("*") if p.suffix.lower() in EXTENSIONES)


def medidas_imagen(ruta: Path) -> dict:
    """Ancho, alto y estadísticas simples de brillo, saturación y nitidez.

    - brillo: media del canal V de HSV.
    - saturacion: media del canal S de HSV.
    - contraste: desviación estándar de la luminancia.
    - nitidez: varianza del Laplaciano (valores bajos indican desenfoque).
    - fondo_v: brillo mediano del borde (aproxima el color del fondo).
    """
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    img = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if img is None:
        return {"legible": False}
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = max(2, int(0.04 * min(h, w)))
    borde = np.concatenate([hsv[:g, :, 2].ravel(), hsv[-g:, :, 2].ravel(),
                            hsv[:, :g, 2].ravel(), hsv[:, -g:, 2].ravel()])
    return {
        "legible": True,
        "ancho": w,
        "alto": h,
        "brillo": float(hsv[..., 2].mean()),
        "saturacion": float(hsv[..., 1].mean()),
        "contraste": float(gris.std()),
        "nitidez": float(cv2.Laplacian(gris, cv2.CV_64F).var()),
        "fondo_v": float(np.median(borde)),
        "md5": hashlib.md5(datos.tobytes()).hexdigest(),
    }


def construir_inventario(raiz: Path, con_medidas: bool = True) -> pd.DataFrame:
    """Recorre el dataset y arma un DataFrame con una fila por imagen.

    Columnas: ruta (relativa a ``raiz``), fruta, calidad, bytes y, si
    ``con_medidas``, las de ``medidas_imagen``.
    """
    raiz = Path(raiz)
    filas = []
    for ruta in listar_imagenes(raiz):
        rel = ruta.relative_to(raiz)
        partes = list(rel.parts[:-1]) + [ruta.stem]
        fila = {
            "ruta": rel.as_posix(),
            "fruta": _buscar_clave(partes, FRUTAS_DATASET),
            "calidad": _buscar_clave(list(rel.parts[:-1]), CALIDADES_DATASET),
            "bytes": ruta.stat().st_size,
        }
        if con_medidas:
            fila.update(medidas_imagen(ruta))
        filas.append(fila)
    return pd.DataFrame(filas)
