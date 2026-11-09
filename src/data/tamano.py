"""Estimación del tamaño de la fruta a partir de su máscara de segmentación.

El área A (píxeles de la máscara) se convierte en el diámetro del círculo de
igual área, d = 2*sqrt(A/pi), y se normaliza por el lado mayor de la imagen
para que no dependa de la resolución. La categoría (pequeño, mediano, grande)
sale de percentiles calculados por fruta sobre el conjunto de entrenamiento.
"""

from __future__ import annotations

import numpy as np

CATEGORIAS_TAMANO = ("pequeno", "mediano", "grande")


def area_mascara(mascara: np.ndarray) -> int:
    """Número de píxeles de la fruta (valores distintos de cero)."""
    return int(np.count_nonzero(mascara))


def diametro_equivalente(area: float) -> float:
    """Diámetro del círculo con la misma área: d = 2*sqrt(A/pi)."""
    if area < 0:
        raise ValueError("El área no puede ser negativa")
    return float(2.0 * np.sqrt(area / np.pi))


def diametro_normalizado(diametro_px: float, forma: tuple[int, ...]) -> float:
    """Divide el diámetro por el lado mayor de la imagen (valor en (0, 1])."""
    return float(diametro_px / max(forma[0], forma[1]))


def calcular_umbrales(diametros: np.ndarray, percentiles=(100 / 3, 200 / 3)) -> tuple[float, float]:
    """Umbrales pequeño/mediano y mediano/grande como percentiles de ``diametros``."""
    diametros = np.asarray(diametros, dtype=float)
    if diametros.size == 0:
        raise ValueError("Se necesita al menos un diámetro para calcular umbrales")
    bajo, alto = np.percentile(diametros, percentiles)
    return float(bajo), float(alto)


def categoria_tamano(diametro: float, umbrales: tuple[float, float]) -> str:
    """Asigna pequeno, mediano o grande según los umbrales (bajo, alto)."""
    bajo, alto = umbrales
    if diametro < bajo:
        return CATEGORIAS_TAMANO[0]
    if diametro < alto:
        return CATEGORIAS_TAMANO[1]
    return CATEGORIAS_TAMANO[2]


def estimar_factor_escala(diametros_px: np.ndarray, diametros_cm: np.ndarray) -> float:
    """Factor k (cm/píxel) por mínimos cuadrados sin intercepto: d_cm ≈ k·d_px.

    k = sum(d_px * d_cm) / sum(d_px^2). Sirve para las fotos propias, que se
    toman a distancia fija y tienen el diámetro real medido con regla.
    """
    x = np.asarray(diametros_px, dtype=float)
    y = np.asarray(diametros_cm, dtype=float)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("Los arreglos deben tener la misma longitud y no estar vacíos")
    return float(np.dot(x, y) / np.dot(x, x))


def estimar_tamano(mascara: np.ndarray, umbrales: tuple[float, float] | None = None) -> dict:
    """Calcula área, diámetro en píxeles, diámetro normalizado y categoría.

    Parameters
    ----------
    mascara : np.ndarray
        Máscara binaria (H, W) de la fruta.
    umbrales : tuple, opcional
        Umbrales de ``calcular_umbrales``. Si es None no se asigna categoría.

    Returns
    -------
    dict
        Llaves ``area_px``, ``diametro_px``, ``diametro_norm`` y ``categoria``.
    """
    area = area_mascara(mascara)
    d_px = diametro_equivalente(area)
    d_norm = diametro_normalizado(d_px, mascara.shape)
    categoria = categoria_tamano(d_norm, umbrales) if umbrales is not None else None
    return {"area_px": area, "diametro_px": d_px, "diametro_norm": d_norm, "categoria": categoria}
