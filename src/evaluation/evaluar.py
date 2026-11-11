"""Métricas, intervalos de confianza y gráficas de evaluación."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from src.config import SEMILLA


def metricas(y_true, y_pred, clases, y_score=None) -> dict:
    """Exactitud, precisión/recall/F1 macro y F1/recall por clase.

    ``y_score`` (probabilidades, columnas en el orden de ``clases``) es opcional
    y se usa para el AUC (uno contra el resto si hay más de dos clases).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {"n": int(len(y_true)),
           "exactitud": accuracy_score(y_true, y_pred),
           "precision_macro": precision_score(y_true, y_pred, labels=clases, average="macro",
                                              zero_division=0),
           "recall_macro": recall_score(y_true, y_pred, labels=clases, average="macro",
                                        zero_division=0),
           "f1_macro": f1_score(y_true, y_pred, labels=clases, average="macro", zero_division=0)}
    f1s = f1_score(y_true, y_pred, labels=clases, average=None, zero_division=0)
    recs = recall_score(y_true, y_pred, labels=clases, average=None, zero_division=0)
    for c, f, r in zip(clases, f1s, recs):
        out[f"f1_{c}"] = f
        out[f"recall_{c}"] = r
    if y_score is not None:
        y_score = np.asarray(y_score)
        if len(clases) == 2:
            out["auc"] = roc_auc_score(y_true == clases[1], y_score[:, 1])
        else:
            # sklearn exige las etiquetas en orden alfabético y las columnas en ese orden
            orden = sorted(clases)
            cols = [list(clases).index(c) for c in orden]
            out["auc"] = roc_auc_score(y_true, y_score[:, cols], multi_class="ovr", labels=orden)
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in out.items()}


def intervalo_bootstrap(y_true, y_pred, funcion=None, n: int = 1000, alfa: float = 0.05,
                        semilla: int = SEMILLA) -> tuple[float, float]:
    """Intervalo de confianza percentil por bootstrap de una métrica (F1 macro por defecto)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if funcion is None:
        def funcion(a, b):
            return f1_score(a, b, average="macro", zero_division=0)
    rng = np.random.default_rng(semilla)
    valores = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        valores[i] = funcion(y_true[idx], y_pred[idx])
    return float(np.quantile(valores, alfa / 2)), float(np.quantile(valores, 1 - alfa / 2))


def metricas_por_grupo(df: pd.DataFrame, grupo: str, clases, col_real="real",
                       col_pred="pred") -> pd.DataFrame:
    """Métricas separadas por grupo (por ejemplo, por fruta)."""
    filas = []
    for g, sub in df.groupby(grupo):
        m = metricas(sub[col_real], sub[col_pred], clases)
        filas.append({grupo: g, **m})
    return pd.DataFrame(filas)


def graficar_matriz_confusion(ax, y_true, y_pred, clases, titulo: str = "") -> None:
    """Matriz de confusión normalizada por fila, con los conteos entre paréntesis."""
    cm = confusion_matrix(y_true, y_pred, labels=clases)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(clases)):
        for j in range(len(clases)):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.6 else "black", fontsize=9)
    ax.set_xticks(range(len(clases)), clases)
    ax.set_yticks(range(len(clases)), clases)
    ax.set_xlabel("Predicha")
    ax.set_ylabel("Real")
    ax.set_title(titulo)


def graficar_roc(ax, y_true, y_score, clases, etiqueta: str) -> None:
    """Curva ROC de la clase positiva ``clases[-1]`` (para dos clases)."""
    y_true = np.asarray(y_true)
    pos = len(clases) - 1
    fpr, tpr, _ = roc_curve(y_true == clases[pos], np.asarray(y_score)[:, pos])
    auc = roc_auc_score(y_true == clases[pos], np.asarray(y_score)[:, pos])
    ax.plot(fpr, tpr, label=f"{etiqueta} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=0.8)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
