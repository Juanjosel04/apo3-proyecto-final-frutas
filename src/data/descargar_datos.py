"""Descarga el dataset Fruit Quality Classification de Kaggle.

Fuente: R. Park, "Fruit Quality Classification", Kaggle.
https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification

Uso:
    python -m src.data.descargar_datos

Se usa ``kagglehub``. Para datasets públicos normalmente no hace falta
token; si Kaggle lo pide, se debe crear ``~/.kaggle/kaggle.json`` (nunca
dentro del repo). Los archivos quedan en ``data/raw/kaggle/`` (ignorado por git).
"""

import shutil
from pathlib import Path

import kagglehub

from src.config import DATA_RAW_KAGGLE, KAGGLE_DATASET


def descargar(destino: Path = DATA_RAW_KAGGLE, forzar: bool = False) -> Path:
    """Descarga el dataset y lo copia a ``destino``.

    Parameters
    ----------
    destino : Path
        Carpeta donde quedan las imágenes.
    forzar : bool
        Si es True, borra lo que haya en ``destino`` y vuelve a copiar.

    Returns
    -------
    Path
        Ruta de la carpeta con el dataset.
    """
    contenido = [p for p in destino.iterdir() if p.name != ".gitkeep"] if destino.exists() else []
    if contenido and not forzar:
        print(f"El dataset ya está en {destino}. Use forzar=True para recopiar.")
        return destino

    ruta_cache = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"Descargado en caché: {ruta_cache}")

    destino.mkdir(parents=True, exist_ok=True)
    for p in contenido:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    for item in ruta_cache.iterdir():
        objetivo = destino / item.name
        if item.is_dir():
            shutil.copytree(item, objetivo)
        else:
            shutil.copy2(item, objetivo)
    print(f"Dataset copiado a {destino}")
    return destino


if __name__ == "__main__":
    descargar()
