"""Utilidades para las fotos propias del equipo (``data/propias``).

El tamaño de estas fotos es **relativo**: se clasifica con los percentiles 33 y
66 del diámetro normalizado del dataset (``experiments/results/umbrales_tamano.json``).
Las fotos no comparten distancia ni resolución, así que no sirven para estimar
el diámetro en cm.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import ETIQUETAS_PROPIAS, FOTOS_PROPIAS, RESULTADOS
from src.data.segmentacion import mascara_fruta
from src.data.tamano import estimar_tamano

UMBRALES_TAMANO = RESULTADOS / "umbrales_tamano.json"


def cargar_umbrales(ruta: Path = UMBRALES_TAMANO) -> dict[str, tuple[float, float]]:
    """Umbrales (p33, p66) del diámetro normalizado por fruta."""
    datos = json.loads(Path(ruta).read_text(encoding="utf8"))
    return {f: (u["p33"], u["p66"]) for f, u in datos.items()}


def tamano_relativo(img_bgr, fruta: str, umbrales: dict | None = None) -> dict:
    """Segmenta la fruta y devuelve su diámetro normalizado y su categoría relativa."""
    umbrales = umbrales or cargar_umbrales()
    mascara = mascara_fruta(img_bgr)
    t = estimar_tamano(mascara, umbrales[fruta])
    return {"diametro_norm": t["diametro_norm"], "categoria": t["categoria"],
            "area_rel": t["area_px"] / mascara.size}


def cargar_etiquetas_propias(ruta: Path = ETIQUETAS_PROPIAS,
                             carpeta: Path = FOTOS_PROPIAS) -> pd.DataFrame:
    """Etiquetas de las fotos propias con la ruta completa de cada imagen."""
    df = pd.read_csv(ruta, dtype={"diametro_cm": float})
    df["ruta"] = [carpeta / f / c / a for f, c, a in zip(df["fruta"], df["calidad"], df["archivo"])]
    return df
