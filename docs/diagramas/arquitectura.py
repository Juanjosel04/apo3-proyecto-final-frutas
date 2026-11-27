"""Genera el diagrama de bloques del sistema (SVG, PDF y PNG).

Tiene el ancho de una columna del informe IEEE. Es la fuente editable:
se cambian los textos de BLOQUES y se vuelve a correr.

Uso (desde la raíz del repo):
    python docs/diagramas/arquitectura.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SALIDA = Path(__file__).resolve().parent

AZUL, VERDE, AMARILLO, ROJO, GRIS = "#dbe9f6", "#dff0e0", "#fdf0d5", "#f6dcdc", "#e3e3e3"

# nombre: (x, y, ancho, alto, color, texto)
BLOQUES = {
    "foto": (2.2, 11.0, 5.6, 0.9, "white", "Foto de una fruta (archivo o cámara)\nlado mayor a 1024 px"),
    "seg": (1.6, 9.3, 6.8, 1.2, AZUL, "Segmentación clásica\nΔE al fondo + Otsu → GrabCut → morfología\nmáscara M (contorno de mayor área)"),
    "tam": (0.0, 6.8, 4.1, 1.8, AMARILLO, "Tamaño relativo\nd = 2·√(A/π) / máx(H, W)\npequeño / mediano / grande\n(percentiles 33 y 66 por fruta)"),
    "rec": (4.6, 7.2, 5.3, 1.1, AMARILLO, "Recorte cuadrado centrado,\nfondo negro (192 × 192 px)"),
    "fruta": (4.6, 5.4, 5.3, 1.1, GRIS, "Fruta: la indica el usuario\no la detecta un Random Forest"),
    "svm": (4.6, 3.3, 2.5, 1.5, ROJO, "78 características\nz-score\nSVM RBF + Platt"),
    "cnn": (7.4, 3.3, 2.5, 1.5, ROJO, "Recorte 128 px\nCNN desde cero\n(alternativa)"),
    "cal": (4.5, 1.6, 5.5, 1.0, VERDE, "Calidad + probabilidad por clase\nrevisión humana si confianza < 0,70"),
    "app": (0.6, 0.0, 9.3, 0.9, "#e9e0f3", "App Streamlit: contorno, recorte, calidad, fruta y tamaño"),
}


def borde(n, lado):
    x, y, w, h, *_ = BLOQUES[n]
    return {"arriba": (x + w / 2, y + h), "abajo": (x + w / 2, y),
            "izq": (x, y + h / 2), "der": (x + w, y + h / 2)}[lado]


def flecha(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, lw=1.0, color="#333333",
                                 shrinkA=1, shrinkB=1))


def dibujar():
    fig, ax = plt.subplots(figsize=(3.5, 4.4))
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.2, 12.1)
    ax.axis("off")

    for x, y, w, h, color, texto in BLOQUES.values():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.15",
                                    fc=color, ec="#333333", lw=0.8))
        ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=5.0,
                color="#1a1a1a", linespacing=1.25)

    flecha(ax, borde("foto", "abajo"), borde("seg", "arriba"))
    # de la máscara salen el tamaño y el recorte
    _, ys = borde("seg", "abajo")
    flecha(ax, (2.2, ys), borde("tam", "arriba"))
    flecha(ax, (7.25, ys), borde("rec", "arriba"))
    flecha(ax, borde("rec", "abajo"), borde("fruta", "arriba"))
    flecha(ax, (5.85, BLOQUES["fruta"][1]), borde("svm", "arriba"))
    flecha(ax, (8.65, BLOQUES["fruta"][1]), borde("cnn", "arriba"))
    flecha(ax, borde("svm", "abajo"), (5.85, BLOQUES["cal"][1] + BLOQUES["cal"][3]))
    flecha(ax, borde("cnn", "abajo"), (8.65, BLOQUES["cal"][1] + BLOQUES["cal"][3]))
    flecha(ax, borde("cal", "abajo"), (7.25, BLOQUES["app"][1] + BLOQUES["app"][3]))
    flecha(ax, borde("tam", "abajo"), (1.9, BLOQUES["app"][1] + BLOQUES["app"][3]))

    for ext in ("svg", "pdf", "png"):
        fig.savefig(SALIDA / f"arquitectura.{ext}", bbox_inches="tight", dpi=400 if ext == "png" else None)
    plt.close(fig)
    print(f"Diagrama guardado en {SALIDA}")


if __name__ == "__main__":
    dibujar()
