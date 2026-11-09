"""Preparación de los datos: tamaño de cada fruta y recortes de ``mixed_quality``.

Uso (desde la raíz del repo, después de descargar el dataset):
    python -m src.data.preprocesamiento

Genera:
- ``data/processed/tamanos_kaggle.csv``: área y diámetro normalizado de una muestra
  estratificada de imágenes individuales (250 por fruta y calidad).
- ``experiments/results/umbrales_tamano.json``: percentiles 33 y 66 del diámetro por fruta.
- ``data/interim/mixed/``: un recorte (imagen + máscara) por fruta detectada en ``mixed_quality``.
- ``data/interim/mixed/recortes_mixed.csv``: índice de recortes con las columnas ``calidad`` y
  ``valido`` vacías. Se llenan a mano: ``valido`` = 0 si el recorte no es una fruta
  completa (fondo, dos frutas juntas, fruta cortada).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import DATA_INTERIM, DATA_PROCESSED, DATA_RAW_KAGGLE, RESULTADOS, SEMILLA
from src.data.segmentacion import mascara_fruta, separar_frutas
from src.data.tamano import calcular_umbrales, estimar_tamano


def _leer_bgr(ruta: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_COLOR)


def _guardar(ruta: Path, img: np.ndarray) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(ruta.suffix, img)
    if ok:
        buf.tofile(str(ruta))


def tamanos_dataset(inventario: pd.DataFrame, raiz: Path = DATA_RAW_KAGGLE,
                    n_por_grupo: int | None = 250) -> pd.DataFrame:
    """Calcula el tamaño de las imágenes individuales (calidad good o bad).

    Con ``n_por_grupo`` se toma una muestra estratificada por fruta y calidad
    (semilla fija). En el dataset no se conoce la distancia de la cámara, así que
    el tamaño es relativo y una muestra basta para estimar los percentiles.
    """
    indiv = inventario[inventario["calidad"].isin(["good", "bad"])]
    if n_por_grupo is not None:
        n = min(n_por_grupo, int(indiv.groupby(["fruta", "calidad"]).size().min()))
        indiv = indiv.groupby(["fruta", "calidad"]).sample(n, random_state=SEMILLA)
    filas = []
    for _, fila in indiv.iterrows():
        img = _leer_bgr(raiz / fila["ruta"])
        m = mascara_fruta(img)
        t = estimar_tamano(m)
        x, y, w, h = cv2.boundingRect(m)
        toca_borde = x == 0 or y == 0 or x + w == m.shape[1] or y + h == m.shape[0]
        filas.append({"ruta": fila["ruta"], "fruta": fila["fruta"], "calidad": fila["calidad"],
                      "area_px": t["area_px"], "diametro_px": t["diametro_px"],
                      "diametro_norm": t["diametro_norm"], "area_rel": t["area_px"] / m.size,
                      "toca_borde": toca_borde})
    return pd.DataFrame(filas)


def umbrales_por_fruta(tamanos: pd.DataFrame) -> dict:
    """Percentiles 33 y 66 del diámetro normalizado por fruta.

    Se excluyen máscaras dudosas (fruta que toca el borde o que ocupa menos del 5 %
    o más del 95 % de la imagen), que suelen ser errores de segmentación.
    """
    validas = tamanos[(~tamanos["toca_borde"]) & tamanos["area_rel"].between(0.05, 0.95)]
    return {f: dict(zip(("p33", "p66"), calcular_umbrales(g["diametro_norm"])))
            for f, g in validas.groupby("fruta")}


def recortes_mixed(inventario: pd.DataFrame, raiz: Path = DATA_RAW_KAGGLE,
                   destino: Path = DATA_INTERIM / "mixed") -> pd.DataFrame:
    """Separa cada fruta de las imágenes mixed y guarda recorte y máscara."""
    mixed = inventario[inventario["calidad"] == "mixed"]
    filas = []
    for _, fila in mixed.iterrows():
        img = _leer_bgr(raiz / fila["ruta"])
        base = Path(fila["ruta"]).stem.replace(" ", "_")
        for k, fr in enumerate(separar_frutas(img)):
            nombre = f"{fila['fruta']}_{base}_{k:02d}"
            _guardar(destino / "imagenes" / f"{nombre}.jpg", fr.imagen)
            _guardar(destino / "mascaras" / f"{nombre}.png", fr.mascara)
            t = estimar_tamano(fr.mascara)
            filas.append({"recorte": f"{nombre}.jpg", "origen": fila["ruta"], "fruta": fila["fruta"],
                          "indice": k, "x": fr.caja[0], "y": fr.caja[1], "ancho": fr.caja[2],
                          "alto": fr.caja[3], "area_px": fr.area,
                          "diametro_norm_origen": 2 * np.sqrt(fr.area / np.pi) / max(img.shape[:2]),
                          "diametro_norm_recorte": t["diametro_norm"], "calidad": "", "valido": ""})
    indice = pd.DataFrame(filas)
    destino.mkdir(parents=True, exist_ok=True)
    indice.to_csv(destino / "recortes_mixed.csv", index=False)
    return indice


def main() -> None:
    inv = pd.read_csv(DATA_PROCESSED / "inventario_kaggle.csv")

    tam = tamanos_dataset(inv)
    tam.to_csv(DATA_PROCESSED / "tamanos_kaggle.csv", index=False)
    umbrales = umbrales_por_fruta(tam)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    (RESULTADOS / "umbrales_tamano.json").write_text(json.dumps(umbrales, indent=2), encoding="utf8")
    print(f"Tamaños de {len(tam)} imágenes. Umbrales: {umbrales}")

    rec = recortes_mixed(inv)
    print(f"{len(rec)} recortes de {rec['origen'].nunique()} imágenes mixed")


if __name__ == "__main__":
    main()
