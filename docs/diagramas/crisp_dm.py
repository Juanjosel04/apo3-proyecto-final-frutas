"""Genera el diagrama CRISP-DM aplicado al proyecto (SVG, PDF y PNG).

Es la fuente editable del diagrama: se cambian los textos de FASES y se
vuelve a correr.

Uso (desde la raíz del repo):
    python docs/diagramas/crisp_dm.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SALIDA = Path(__file__).resolve().parent

FASES = [
    ("1. Comprensión del negocio", "#dbe9f6", [
        "Pérdidas poscosecha y selección manual",
        "Salidas: calidad (buena/regular/mala) y tamaño",
        "Éxito: F1 macro, recall de 'mala', < 2 s",
        "Ética e impactos desde el inicio",
    ]),
    ("2. Comprensión de los datos", "#dff0e0", [
        "EDA del dataset de Kaggle: frutas y clases",
        "Desbalance, resolución, brillo, fondos",
        "mixed_quality: varias frutas por foto",
        "Elección de frutas y definición de clases",
    ]),
    ("3. Preparación de los datos", "#fdf0d5", [
        "Segmentación: ΔE en Lab + Otsu + morfología",
        "Recorte de cada fruta (watershed si se tocan)",
        "Tamaño: d = 2·√(A/π), normalizado",
        "Características de color, textura y forma",
    ]),
    ("4. Modelado", "#f6dcdc", [
        "Línea base: regresión logística",
        "SVM (RBF) y Random Forest + GridSearchCV",
        "CNN simple desde cero (Keras)",
        "Pesos por clase y aumentación",
    ]),
    ("5. Evaluación", "#e9e0f3", [
        "Test interno estratificado (Kaggle)",
        "Test externo: fotos propias del equipo",
        "Matriz de confusión, curvas, sobreajuste",
        "Error del tamaño contra el diámetro en cm",
    ]),
    ("6. Despliegue", "#e3e3e3", [
        "App Streamlit: cargar imagen o cámara",
        "Muestra calidad, confianza y tamaño",
        "Aviso de revisión humana si la confianza es baja",
        "Tiempo de inferencia medido",
    ]),
]


ANCHO, ALTO = 5.2, 2.5
# esquina inferior izquierda de cada fase, en forma de anillo
POS = [(0.3, 5.0), (6.4, 5.0), (12.5, 5.0), (12.5, 1.2), (6.4, 1.2), (0.3, 1.2)]
FS_TEXTO = 7.8


def caja(ax, x, y, titulo, color, lineas):
    ax.add_patch(FancyBboxPatch((x, y), ANCHO, ALTO, boxstyle="round,pad=0.05,rounding_size=0.18",
                                fc=color, ec="#333333", lw=1.1))
    ax.text(x + ANCHO / 2, y + ALTO - 0.33, titulo, ha="center", va="center",
            fontsize=10, fontweight="bold", color="#1a1a1a")
    for i, linea in enumerate(lineas):
        ax.text(x + 0.18, y + ALTO - 0.82 - i * 0.42, "• " + linea, ha="left", va="center",
                fontsize=FS_TEXTO, color="#222222")


def fuente(ax, x, y, texto, ancho=5.2, alto=0.8):
    ax.add_patch(FancyBboxPatch((x, y), ancho, alto, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc="white", ec="#4a6b8a", lw=1.0))
    ax.text(x + ancho / 2, y + alto / 2, texto, ha="center", va="center", fontsize=FS_TEXTO,
            color="#1f3b57")


def flecha(ax, p0, p1, ls="-", color="#333333", rad=0.0, texto=None, xy_texto=None):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, lw=1.3,
                                 ls=ls, color=color, connectionstyle=f"arc3,rad={rad}"))
    if texto:
        xt, yt = xy_texto if xy_texto else ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        ax.text(xt, yt, texto, ha="center", va="center", fontsize=7.3, style="italic", color=color,
                bbox=dict(fc="white", ec="none", pad=1.0))


def dibujar():
    fig, ax = plt.subplots(figsize=(14, 7.6))
    ax.set_xlim(0, 18.0)
    ax.set_ylim(-0.6, 10.0)
    ax.axis("off")

    for (x, y), (titulo, color, lineas) in zip(POS, FASES):
        caja(ax, x, y, titulo, color, lineas)

    der = lambda i: POS[i][0] + ANCHO  # noqa: E731
    izq = lambda i: POS[i][0]  # noqa: E731
    cx = lambda i: POS[i][0] + ANCHO / 2  # noqa: E731
    sup = lambda i: POS[i][1] + ALTO  # noqa: E731
    inf = lambda i: POS[i][1]  # noqa: E731

    # Flujo principal
    flecha(ax, (der(0), 6.75), (izq(1), 6.75))
    flecha(ax, (der(1), 6.75), (izq(2), 6.75))
    flecha(ax, (cx(2), inf(2)), (cx(3), sup(3)))
    flecha(ax, (izq(3), 2.95), (der(4), 2.95))
    flecha(ax, (izq(4), 2.95), (der(5), 2.95))

    # Iteraciones (punteadas)
    rojo = "#a33b3b"
    flecha(ax, (izq(1), 5.55), (der(0), 5.55), ls="--", color=rojo,
           texto="ajustar\nclases", xy_texto=((izq(1) + der(0)) / 2, 6.1))
    flecha(ax, (der(4), 1.65), (izq(3), 1.65), ls="--", color=rojo,
           texto="reajustar\nmodelos", xy_texto=((izq(3) + der(4)) / 2, 2.2))
    flecha(ax, (cx(4) + 1.2, sup(4)), (izq(2) + 0.9, inf(2)), ls="--", color=rojo, rad=-0.2,
           texto="errores de segmentación o de características", xy_texto=(11.6, 4.35))
    flecha(ax, (cx(5), sup(5)), (cx(0), inf(0)), ls="--", color=rojo,
           texto="¿cumple los criterios de éxito?", xy_texto=(cx(0), 4.35))

    # Fuentes de datos
    azul = "#4a6b8a"
    fuente(ax, izq(1), 8.15, "Kaggle: Fruit Quality Classification\n(imágenes individuales por calidad)")
    fuente(ax, izq(2), 8.15, "Imágenes mixed_quality\n(varias frutas por foto, se separan)")
    fuente(ax, izq(4), -0.45, "Fotos propias del equipo\n(etiquetas de calidad + diámetro en cm)")
    flecha(ax, (cx(1), 8.15), (cx(1), sup(1)), color=azul)
    flecha(ax, (cx(2), 8.15), (cx(2), sup(2)), color=azul)
    flecha(ax, (cx(4), 0.35), (cx(4), inf(4)), color=azul)
    ax.text(cx(4) + 0.15, 0.78, "prueba externa", ha="left", va="center", fontsize=7.3,
            style="italic", color=azul)
    ax.text(izq(1) - 0.45, 8.55, "Fuentes\nde datos", ha="right", va="center", fontsize=8.5,
            fontweight="bold", color=azul)

    ax.text(9.0, 9.6, "CRISP-DM aplicado: clasificación de calidad y tamaño de frutas",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(0.3, 0.0, "Continuas: flujo principal\nPunteadas: iteraciones entre fases",
            ha="left", va="center", fontsize=7.3, color="#555555")

    for ext in ("svg", "pdf", "png"):
        fig.savefig(SALIDA / f"crisp_dm.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    dibujar()
    print(f"Diagrama guardado en {SALIDA}")
