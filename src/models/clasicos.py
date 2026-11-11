"""Modelos de ML clásico sobre características manuales y sus rejillas de búsqueda.

Todos van dentro de un ``Pipeline`` con ``StandardScaler`` para que la
estandarización se ajuste solo con los pliegues de entrenamiento de cada
validación cruzada (sin fuga de información). Se usa ``class_weight="balanced"``
para que el mismo código sirva cuando se agregue la clase "regular", que
quedará minoritaria.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import SEMILLA

NOMBRES = {"logreg": "Regresión logística", "svm": "SVM (RBF)", "rf": "Random Forest"}


def crear_modelo(nombre: str) -> Pipeline:
    """Pipeline (escalado + clasificador) para ``logreg``, ``svm`` o ``rf``."""
    if nombre == "logreg":
        clf = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEMILLA)
    elif nombre == "svm":
        clf = SVC(kernel="rbf", class_weight="balanced", random_state=SEMILLA)
    elif nombre == "rf":
        # El RF no necesita escalado, pero se deja el mismo pipeline por uniformidad
        clf = RandomForestClassifier(class_weight="balanced", n_jobs=1, random_state=SEMILLA)
    else:
        raise ValueError(f"Modelo desconocido: {nombre}")
    return Pipeline([("escalado", StandardScaler()), ("clf", clf)])


def rejilla(nombre: str) -> dict:
    """Hiperparámetros que explora ``GridSearchCV`` para cada modelo."""
    if nombre == "logreg":
        return {"clf__C": [0.01, 0.1, 1, 10, 100]}
    if nombre == "svm":
        return {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.001, 0.01, 0.1]}
    if nombre == "rf":
        return {"clf__n_estimators": [200, 500], "clf__max_depth": [None, 12],
                "clf__min_samples_leaf": [1, 4], "clf__max_features": ["sqrt", 0.3]}
    raise ValueError(f"Modelo desconocido: {nombre}")
