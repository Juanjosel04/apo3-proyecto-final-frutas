"""Recortes segmentados de las frutas del proyecto y partición train/val/test.

Cada imagen individual (good o bad) de ``FRUTAS_PROYECTO`` se segmenta con
GrabCut, se recorta a un cuadrado alrededor de la fruta, se pinta el fondo de
negro y se guarda a ``LADO_RECORTE`` px. Así los modelos solo ven la fruta y
todas las imágenes quedan con la misma resolución, lo que reduce el sesgo de
fondo y de nitidez encontrado en el EDA.

Uso (desde la raíz del repo):
    python -m src.data.recortes

Genera ``data/interim/recortes/`` (no versionado) y ``data/processed/recortes_kaggle.csv``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from src.config import (CALIDAD_DESDE_DATASET, CLASES_ENTRENAMIENTO, DATA_PROCESSED,
                        DATA_RAW_KAGGLE, ETIQUETAS_REGULAR, FRACCION_TEST, FRACCION_VAL,
                        FRUTAS_PROYECTO, LADO_RECORTE, RECORTES, SEMILLA)
from src.data.segmentacion import mascara_fruta
from src.data.tamano import estimar_tamano

INDICE_RECORTES = DATA_PROCESSED / "recortes_kaggle.csv"

# Una máscara con menos del 2 % o más del 98 % de la imagen se toma como fallo
AREA_REL_MIN, AREA_REL_MAX = 0.02, 0.98


def leer_bgr(ruta: Path) -> np.ndarray:
    """Lee una imagen con rutas que tengan espacios o tildes (Windows)."""
    return cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_COLOR)


def guardar_imagen(ruta: Path, img: np.ndarray) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(ruta.suffix, img)
    if ok:
        buf.tofile(str(ruta))


def recorte_cuadrado(img_bgr: np.ndarray, mascara: np.ndarray, lado: int = LADO_RECORTE,
                     margen: float = 0.06) -> tuple[np.ndarray, np.ndarray]:
    """Recorte cuadrado centrado en la fruta, con el fondo en negro.

    El cuadrado mide el lado mayor de la caja envolvente más un margen; lo que
    queda por fuera de la imagen se rellena con ceros. Se conserva la proporción
    de la fruta (no se deforma al redimensionar).

    Returns
    -------
    tuple
        Imagen BGR y máscara (0/255), ambas de ``lado`` x ``lado``.
    """
    x, y, w, h = cv2.boundingRect(mascara)
    if w == 0 or h == 0:
        vacio = np.zeros((lado, lado), np.uint8)
        return np.zeros((lado, lado, 3), np.uint8), vacio
    s = int(round(max(w, h) * (1 + 2 * margen)))
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))

    img = img_bgr.copy()
    img[mascara == 0] = 0
    lienzo = np.zeros((s, s, 3), np.uint8)
    lienzo_m = np.zeros((s, s), np.uint8)
    H, W = mascara.shape
    ax0, ay0 = max(0, x0), max(0, y0)
    ax1, ay1 = min(W, x0 + s), min(H, y0 + s)
    lienzo[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = img[ay0:ay1, ax0:ax1]
    lienzo_m[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = mascara[ay0:ay1, ax0:ax1]

    interp = cv2.INTER_AREA if s > lado else cv2.INTER_LINEAR
    rec = cv2.resize(lienzo, (lado, lado), interpolation=interp)
    rec_m = cv2.resize(lienzo_m, (lado, lado), interpolation=cv2.INTER_NEAREST)
    return rec, rec_m


def _procesar(ruta_rel: str, raiz: Path, destino: Path) -> dict:
    img = leer_bgr(raiz / ruta_rel)
    m = mascara_fruta(img)
    t = estimar_tamano(m)
    x, y, w, h = cv2.boundingRect(m)
    toca_borde = bool(x == 0 or y == 0 or x + w == m.shape[1] or y + h == m.shape[0])
    rec, rec_m = recorte_cuadrado(img, m)

    nombre = Path(ruta_rel).with_suffix("")
    ruta_img = Path("imagenes") / nombre.parent.name / f"{nombre.name}.png"
    ruta_msk = Path("mascaras") / nombre.parent.name / f"{nombre.name}.png"
    guardar_imagen(destino / ruta_img, rec)
    guardar_imagen(destino / ruta_msk, rec_m)
    area_rel = t["area_px"] / m.size
    return {"ruta": ruta_rel, "recorte": ruta_img.as_posix(), "mascara": ruta_msk.as_posix(),
            "area_rel": area_rel, "diametro_norm": t["diametro_norm"], "toca_borde": toca_borde,
            "seg_ok": bool(AREA_REL_MIN <= area_rel <= AREA_REL_MAX)}


def asignar_particion(df: pd.DataFrame, semilla: int = SEMILLA) -> pd.Series:
    """Partición estratificada por fruta y calidad del dataset (70/15/15).

    Se estratifica con la calidad original (good/bad) y no con la del proyecto,
    para que la partición no cambie cuando se etiquete la clase "regular".
    """
    estrato = df["fruta"] + "_" + df["calidad_dataset"]
    idx = np.arange(len(df))
    idx_resto, idx_test = train_test_split(idx, test_size=FRACCION_TEST, stratify=estrato,
                                           random_state=semilla)
    val_rel = FRACCION_VAL / (1 - FRACCION_TEST)
    idx_train, idx_val = train_test_split(idx_resto, test_size=val_rel,
                                          stratify=estrato.iloc[idx_resto], random_state=semilla)
    part = np.empty(len(df), dtype=object)
    part[idx_train], part[idx_val], part[idx_test] = "train", "val", "test"
    return pd.Series(part, index=df.index, name="particion")


def construir_recortes(inventario: pd.DataFrame, frutas=FRUTAS_PROYECTO,
                       raiz: Path = DATA_RAW_KAGGLE, destino: Path = RECORTES,
                       n_jobs: int = -2) -> pd.DataFrame:
    """Segmenta y recorta en paralelo las imágenes good/bad de ``frutas``."""
    sel = inventario[inventario["fruta"].isin(frutas) & inventario["calidad"].isin(["good", "bad"])]
    sel = sel.reset_index(drop=True)
    filas = Parallel(n_jobs=n_jobs, batch_size=16)(
        delayed(_procesar)(r, raiz, destino) for r in sel["ruta"])
    df = pd.DataFrame(filas)
    df.insert(1, "fruta", sel["fruta"].values)
    df.insert(2, "calidad_dataset", sel["calidad"].values)
    df["particion"] = asignar_particion(df)
    return df


def cargar_conjunto(clases=CLASES_ENTRENAMIENTO, solo_seg_ok: bool = True,
                    indice: Path = INDICE_RECORTES) -> pd.DataFrame:
    """Índice de recortes con la etiqueta de calidad del proyecto.

    La calidad sale de ``CALIDAD_DESDE_DATASET`` (good -> buena, bad -> mala). Si
    existe ``ETIQUETAS_REGULAR`` (columnas ``ruta`` y ``calidad``), sus filas
    reemplazan la etiqueta; así se incorpora la clase "regular" sin tocar el código.
    Se devuelven solo las filas cuya calidad está en ``clases``.
    """
    df = pd.read_csv(indice)
    df["calidad"] = df["calidad_dataset"].map(CALIDAD_DESDE_DATASET)
    if Path(ETIQUETAS_REGULAR).exists():
        nuevas = pd.read_csv(ETIQUETAS_REGULAR).set_index("ruta")["calidad"]
        reemplazo = df["ruta"].map(nuevas)
        df.loc[reemplazo.notna(), "calidad"] = reemplazo[reemplazo.notna()]
    if solo_seg_ok:
        df = df[df["seg_ok"]]
    return df[df["calidad"].isin(clases)].reset_index(drop=True)


def main() -> None:
    inv = pd.read_csv(DATA_PROCESSED / "inventario_kaggle.csv")
    df = construir_recortes(inv)
    df.to_csv(INDICE_RECORTES, index=False)
    print(f"{len(df)} recortes; segmentación dudosa: {(~df['seg_ok']).sum()}")
    print(df.groupby(["fruta", "calidad_dataset", "particion"]).size().unstack())


if __name__ == "__main__":
    main()
