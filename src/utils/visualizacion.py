"""Utilidades de visualización compartidas por los notebooks."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config import FIGURAS


def estilo():
    """Estilo común de las figuras (fuente legible al tamaño de columna IEEE)."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    })


def guardar_figura(fig, nombre: str, carpeta: Path = FIGURAS, raster: bool = False) -> None:
    """Guarda la figura en PNG, PDF y SVG (las dos últimas son vectoriales).

    Con ``raster=True`` (mosaicos de fotografías) solo se guarda PNG a 150 dpi:
    las fotos son mapas de bits y en PDF/SVG solo aumentan el peso del repo.
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    if raster:
        fig.savefig(carpeta / f"{nombre}.png", bbox_inches="tight", dpi=150)
        return
    for ext in ("png", "pdf", "svg"):
        fig.savefig(carpeta / f"{nombre}.{ext}", bbox_inches="tight")


def leer_rgb(ruta) -> np.ndarray:
    """Lee una imagen en RGB (OpenCV lee en BGR). Soporta rutas con tildes en Windows."""
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    img = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo leer {ruta}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def mostrar_rejilla(imagenes, titulos=None, columnas=6, tam=1.6, fig=None):
    """Muestra una lista de imágenes RGB en una rejilla y devuelve la figura."""
    filas = int(np.ceil(len(imagenes) / columnas))
    if fig is None:
        fig = plt.figure(figsize=(columnas * tam, filas * tam))
    for i, img in enumerate(imagenes):
        ax = fig.add_subplot(filas, columnas, i + 1)
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.axis("off")
        if titulos is not None:
            ax.set_title(titulos[i], fontsize=7)
    fig.tight_layout()
    return fig
