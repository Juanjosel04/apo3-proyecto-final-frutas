"""Entrenamiento de los modelos de calidad.

Uso (desde la raíz del repo):
    python -m src.training.entrenar clasicos   # LogReg, SVM y RF con GridSearchCV
    python -m src.training.entrenar sesgo      # cuánto predicen la calidad las condiciones de captura
    python -m src.training.entrenar cnn --variante base    # CNN desde cero (Keras)
    python -m src.training.entrenar cnn --variante doble

Protocolo común:
- Partición fija 70/15/15 de ``src.data.recortes`` (estratificada por fruta y calidad).
- Modelos clásicos: ``GridSearchCV`` con validación cruzada estratificada de 5
  pliegues sobre train+val; la prueba (test) solo se usa al final.
- CNN: train para ajustar, val para *early stopping*, test al final.
- Métrica de selección: F1 macro.
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (CHECKPOINTS, CLASES_ENTRENAMIENTO, DATA_PROCESSED, FRUTAS_PROYECTO,
                        RESULTADOS, SEMILLA)
from src.evaluation.evaluar import intervalo_bootstrap, metricas
from src.features.extraccion import CARACTERISTICAS_KAGGLE, columnas_por_grupo
from src.models.clasicos import crear_modelo, rejilla

MODELOS_CLASICOS = ("logreg", "svm", "rf")
# global: un modelo para todas las frutas, sin decirle cuál es
# global_fruta: un modelo con la fruta como variable (one-hot)
# por_fruta: un modelo independiente por fruta
CONFIGURACIONES = ("global", "global_fruta", "por_fruta")


def cargar_caracteristicas(clases=CLASES_ENTRENAMIENTO) -> tuple[pd.DataFrame, list[str]]:
    """Tabla de características con la columna ``calidad`` del proyecto."""
    from src.data.recortes import cargar_conjunto

    feats = pd.read_csv(CARACTERISTICAS_KAGGLE)
    etiquetas = cargar_conjunto(clases)[["ruta", "calidad"]]
    df = feats.merge(etiquetas, on="ruta", how="inner")
    columnas = [c for g in columnas_por_grupo(feats.columns).values() for c in g]
    return df, columnas


def matriz(df: pd.DataFrame, columnas: list[str], con_fruta: bool) -> pd.DataFrame:
    X = df[columnas].copy()
    if con_fruta:
        for f in FRUTAS_PROYECTO:
            X[f"fruta_{f}"] = (df["fruta"] == f).astype(float)
    return X


def _cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)


def _buscar(nombre: str, X, y) -> GridSearchCV:
    gs = GridSearchCV(crear_modelo(nombre), rejilla(nombre), scoring="f1_macro", cv=_cv(),
                      n_jobs=-2, return_train_score=True, refit=True)
    gs.fit(X, y)
    return gs


def _fila_cv(gs: GridSearchCV) -> dict:
    i = gs.best_index_
    r = gs.cv_results_
    return {"mejores_parametros": json.dumps({k.replace("clf__", ""): v
                                              for k, v in gs.best_params_.items()}),
            "cv_f1": r["mean_test_score"][i], "cv_f1_std": r["std_test_score"][i],
            "train_f1": r["mean_train_score"][i]}


def _guardar_cv(gs: GridSearchCV, nombre: str) -> None:
    carpeta = RESULTADOS / "clasicos_cv"
    carpeta.mkdir(parents=True, exist_ok=True)
    r = pd.DataFrame(gs.cv_results_)
    cols = [c for c in r.columns if c.startswith("param_")] + [
        "mean_test_score", "std_test_score", "mean_train_score", "std_train_score",
        "mean_fit_time", "rank_test_score"]
    r[cols].to_csv(carpeta / f"{nombre}.csv", index=False)


def _ajustar(nombre: str, X, y, etiqueta: str):
    """GridSearchCV + guardado. La SVM se calibra después para tener probabilidades.

    La SVM no da probabilidades directamente; se usa ``CalibratedClassifierCV``
    (escalado de Platt con validación cruzada) sobre el mejor modelo, en lugar de
    ``SVC(probability=True)``, que es más lento dentro de la búsqueda.
    """
    gs = _buscar(nombre, X, y)
    _guardar_cv(gs, etiqueta)
    modelo = gs.best_estimator_
    if nombre == "svm":
        modelo = CalibratedClassifierCV(clone(modelo), method="sigmoid", cv=_cv(),
                                        ensemble=False).fit(X, y)
    joblib.dump(modelo, CHECKPOINTS / f"clasico_{etiqueta}.joblib")
    return modelo, _fila_cv(gs)


def _probabilidades(modelo, X, clases) -> np.ndarray:
    """Probabilidades con las columnas en el orden de ``clases``."""
    orden = [list(modelo.classes_).index(c) for c in clases]
    return modelo.predict_proba(X)[:, orden]


def entrenar_clasicos() -> pd.DataFrame:
    """Busca hiperparámetros de los tres modelos en las tres configuraciones."""
    df, columnas = cargar_caracteristicas()
    clases = CLASES_ENTRENAMIENTO
    dev = df[df["particion"] != "test"].reset_index(drop=True)
    test = df[df["particion"] == "test"].reset_index(drop=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    resumen, pruebas = [], []
    pred = test[["ruta", "fruta", "calidad"]].rename(columns={"calidad": "real"})
    for config in CONFIGURACIONES:
        for nombre in MODELOS_CLASICOS:
            t0 = time.time()
            prob = np.zeros((len(test), len(clases)))
            if config == "por_fruta":
                for f in FRUTAS_PROYECTO:
                    d, t = dev[dev["fruta"] == f], (test["fruta"] == f).to_numpy()
                    modelo, fila = _ajustar(nombre, matriz(d, columnas, False), d["calidad"],
                                            f"{config}_{nombre}_{f}")
                    resumen.append({"configuracion": config, "modelo": nombre, "fruta": f, **fila})
                    prob[t] = _probabilidades(modelo, matriz(test[t], columnas, False), clases)
            else:
                con_fruta = config == "global_fruta"
                modelo, fila = _ajustar(nombre, matriz(dev, columnas, con_fruta), dev["calidad"],
                                        f"{config}_{nombre}")
                resumen.append({"configuracion": config, "modelo": nombre, "fruta": "todas", **fila})
                prob = _probabilidades(modelo, matriz(test, columnas, con_fruta), clases)
            y_pred = np.array(clases)[prob.argmax(axis=1)]
            m = metricas(test["calidad"], y_pred, clases, prob)
            lo, hi = intervalo_bootstrap(test["calidad"], y_pred)
            pruebas.append({"configuracion": config, "modelo": nombre, **m,
                            "f1_macro_ic_inf": round(lo, 4), "f1_macro_ic_sup": round(hi, 4),
                            "tiempo_busqueda_s": round(time.time() - t0, 1)})
            pred[f"{config}_{nombre}"] = y_pred
            pred[f"{config}_{nombre}_prob_{clases[-1]}"] = prob[:, -1].round(5)
            print(f"{config:13s} {nombre:7s} test F1 macro = {m['f1_macro']:.4f} "
                  f"({time.time() - t0:.0f} s)")

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(resumen)
    res.round(4).to_csv(RESULTADOS / "clasicos_cv_resumen.csv", index=False)
    tabla = pd.DataFrame(pruebas)
    tabla.to_csv(RESULTADOS / "clasicos_test.csv", index=False)
    pred.to_csv(RESULTADOS / "predicciones_test_clasicos.csv", index=False)
    return tabla


def clasificador_fruta() -> dict:
    """Random Forest que reconoce la fruta a partir de las mismas características.

    Lo necesita la configuración ``global_fruta`` y el estimador de tamaño (los
    umbrales de tamaño son por fruta) cuando la app recibe una foto sin decir la fruta.
    """
    df, columnas = cargar_caracteristicas()
    dev, test = df[df["particion"] != "test"], df[df["particion"] == "test"]
    modelo = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-2, random_state=SEMILLA))
    cv = cross_validate(modelo, dev[columnas], dev["fruta"], cv=_cv(), scoring="f1_macro")
    modelo.fit(dev[columnas], dev["fruta"])
    y_pred = modelo.predict(test[columnas])
    m = metricas(test["fruta"], y_pred, FRUTAS_PROYECTO)
    m["cv_f1"] = round(float(cv["test_score"].mean()), 4)
    joblib.dump(modelo, CHECKPOINTS / "clasificador_fruta_rf.joblib")
    (RESULTADOS / "clasificador_fruta.json").write_text(json.dumps(m, indent=2), encoding="utf8")
    print(f"Clasificador de fruta: test F1 macro = {m['f1_macro']:.4f}")
    return m


def sesgo_captura() -> pd.DataFrame:
    """Qué tanto se puede predecir la calidad solo con las condiciones de captura.

    Usa variables de la imagen completa (brillo, saturación, contraste, nitidez,
    brillo del fondo y resolución), sin mirar la fruta. Si estas variables
    predicen bien la calidad, un modelo con la imagen completa podría aprender
    ese atajo; sirve de referencia para interpretar los resultados.
    """
    inv = pd.read_csv(DATA_PROCESSED / "inventario_kaggle.csv").drop(columns="calidad")
    df, _ = cargar_caracteristicas()
    df = df[["ruta", "fruta", "calidad", "particion"]].merge(inv, on=["ruta", "fruta"])
    variables = ["brillo", "saturacion", "contraste", "nitidez", "fondo_v", "ancho", "alto"]
    dev = df[df["particion"] != "test"]
    filas = []
    modelos = {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
               "rf": RandomForestClassifier(n_estimators=300, n_jobs=-2, random_state=SEMILLA)}
    for alcance in ["todas", *FRUTAS_PROYECTO]:
        d = dev if alcance == "todas" else dev[dev["fruta"] == alcance]
        for nombre, modelo in modelos.items():
            cv = cross_validate(modelo, d[variables], d["calidad"], cv=_cv(), scoring="f1_macro")
            filas.append({"fruta": alcance, "modelo": nombre,
                          "cv_f1": round(float(cv["test_score"].mean()), 4),
                          "cv_f1_std": round(float(cv["test_score"].std()), 4)})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RESULTADOS / "sesgo_captura.csv", index=False)
    print(tabla)
    return tabla


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("que", choices=["clasicos", "sesgo", "cnn"])
    parser.add_argument("--epocas", type=int, default=60)
    parser.add_argument("--variante", default="doble", help="variante de la CNN: base o doble")
    args = parser.parse_args()
    if args.que == "clasicos":
        print(entrenar_clasicos())
        clasificador_fruta()
    elif args.que == "sesgo":
        sesgo_captura()
    else:
        from src.models.cnn import entrenar_cnn
        entrenar_cnn(epocas=args.epocas, variante=args.variante)


if __name__ == "__main__":
    main()
