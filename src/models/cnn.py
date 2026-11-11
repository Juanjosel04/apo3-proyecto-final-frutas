"""CNN simple entrenada desde cero (Keras) para la calidad de la fruta.

Arquitectura (entrada 128x128x3, recorte con el fondo en negro):
    aumentación (volteos, rotación, zoom; solo en entrenamiento)
    3 bloques [(Conv2D 3x3 -> BatchNorm -> ReLU) x k -> MaxPool 2x2] con 32, 64 y 128 filtros
    GlobalAveragePooling -> Dropout -> Dense(64, ReLU) -> Dropout -> Dense(softmax)

Variantes (``VARIANTES``): ``base`` con k = 1 convolución por bloque y ``doble``
con k = 2. La primera subajustó (F1 de entrenamiento y validación casi iguales,
~0,94), por eso se probó la segunda con más capacidad.

Se usa *global average pooling* en vez de ``Flatten`` porque reduce la capa
densa de ~2,1 M a ~8 mil parámetros, lo que limita el sobreajuste con unas
4.500 imágenes de entrenamiento.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.config import (CHECKPOINTS, CLASES_ENTRENAMIENTO, RECORTES, RESULTADOS, SEMILLA,
                        TAM_IMAGEN)

VARIANTES = {
    "base": {"convs_por_bloque": 1, "paciencia": 10},
    "doble": {"convs_por_bloque": 2, "paciencia": 12},
}
VARIANTE_FINAL = "doble"


def rutas_cnn(variante: str = VARIANTE_FINAL) -> dict:
    """Rutas del modelo y de los resultados de una variante."""
    return {"modelo": CHECKPOINTS / f"cnn_calidad_{variante}.keras",
            "resultados": RESULTADOS / f"cnn_{variante}_resultados.json",
            "historial": RESULTADOS / f"cnn_{variante}_historial.csv",
            "predicciones": RESULTADOS / f"predicciones_test_cnn_{variante}.csv"}


MODELO_CNN = rutas_cnn()["modelo"]


def cargar_imagenes(rutas, lado: int = TAM_IMAGEN[0]) -> np.ndarray:
    """Lee los recortes, los pasa a RGB y los reduce a ``lado`` x ``lado`` (uint8)."""
    import cv2

    from src.data.recortes import leer_bgr

    salida = np.empty((len(rutas), lado, lado, 3), np.uint8)
    for i, r in enumerate(rutas):
        img = leer_bgr(RECORTES / r)
        img = cv2.resize(img, (lado, lado), interpolation=cv2.INTER_AREA)
        salida[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return salida


def capas_aumentacion():
    """Volteos, rotación libre y zoom leve. El relleno es negro, igual que el fondo."""
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=SEMILLA),
        layers.RandomRotation(0.5, fill_mode="constant", fill_value=0.0, seed=SEMILLA),
        layers.RandomZoom((-0.1, 0.1), fill_mode="constant", fill_value=0.0, seed=SEMILLA),
    ], name="aumentacion")


def construir_cnn(n_clases: int, entrada=(*TAM_IMAGEN, 3), filtros=(32, 64, 128),
                  convs_por_bloque: int = 1, dropout: float = 0.4):
    """Construye y compila la CNN."""
    from tensorflow import keras
    from tensorflow.keras import layers

    x_in = keras.Input(shape=entrada)
    x = capas_aumentacion()(x_in)
    x = layers.Rescaling(1.0 / 255)(x)
    for f in filtros:
        for _ in range(convs_por_bloque):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout / 2)(x)
    salida = layers.Dense(n_clases, activation="softmax")(x)
    modelo = keras.Model(x_in, salida, name="cnn_calidad")
    modelo.compile(optimizer=keras.optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


def entrenar_cnn(epocas: int = 60, lote: int = 32, clases=CLASES_ENTRENAMIENTO,
                 variante: str = VARIANTE_FINAL) -> dict:
    """Entrena una variante de la CNN con train, detiene con val y evalúa en test.

    Guarda el modelo, el historial por época, las métricas y las predicciones de test.
    """
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow import keras

    from src.data.recortes import cargar_conjunto
    from src.evaluation.evaluar import intervalo_bootstrap, metricas

    keras.utils.set_random_seed(SEMILLA)
    df = cargar_conjunto(clases)
    y_todo = df["calidad"].map({c: i for i, c in enumerate(clases)}).to_numpy()
    X = cargar_imagenes(df["recorte"])
    part = {p: (df["particion"] == p).to_numpy() for p in ("train", "val", "test")}

    pesos = compute_class_weight("balanced", classes=np.arange(len(clases)),
                                 y=y_todo[part["train"]])
    ds_train = (tf.data.Dataset.from_tensor_slices((X[part["train"]], y_todo[part["train"]]))
                .shuffle(4096, seed=SEMILLA).batch(lote).prefetch(tf.data.AUTOTUNE))
    ds_val = tf.data.Dataset.from_tensor_slices((X[part["val"]], y_todo[part["val"]])).batch(lote)

    conf = VARIANTES[variante]
    rutas = rutas_cnn(variante)
    modelo = construir_cnn(len(clases), convs_por_bloque=conf["convs_por_bloque"])
    # Respaldo por época: si el entrenamiento se interrumpe, al relanzarlo continúa
    # desde la última época guardada. El historial se va escribiendo en un CSV para
    # no perder las épocas anteriores a la interrupción. (Limitación: el contador de
    # paciencia de EarlyStopping se reinicia al reanudar.)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    respaldo = CHECKPOINTS / f"respaldo_cnn_{variante}"
    log_epocas = CHECKPOINTS / f"cnn_{variante}_epocas.csv"
    if not respaldo.exists() and log_epocas.exists():
        log_epocas.unlink()  # entrenamiento nuevo
    callbacks = [
        keras.callbacks.BackupAndRestore(str(respaldo)),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=conf["paciencia"],
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
        # al final, para que registre la tasa de aprendizaje que agrega ReduceLROnPlateau
        keras.callbacks.CSVLogger(str(log_epocas), append=True),
    ]
    t0 = time.time()
    modelo.fit(ds_train, validation_data=ds_val, epochs=epocas, verbose=2,
               class_weight=dict(enumerate(pesos)), callbacks=callbacks)
    duracion = time.time() - t0

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    modelo.save(rutas["modelo"])
    historial = pd.read_csv(log_epocas).drop(columns="epoch").reset_index(drop=True)
    historial.index.name = "epoca"
    historial.round(5).to_csv(rutas["historial"])

    resultado = {"variante": variante, "parametros": int(modelo.count_params()), "epocas": len(historial),
                 "mejor_epoca": int(historial["val_loss"].idxmin()) + 1,
                 "tiempo_entrenamiento_s": round(duracion, 1)}
    pred = df.loc[part["test"], ["ruta", "fruta", "calidad"]].rename(columns={"calidad": "real"})
    for p in ("train", "val", "test"):
        prob = modelo.predict(X[part[p]], batch_size=128, verbose=0)
        y_pred = np.array(clases)[prob.argmax(axis=1)]
        resultado[p] = metricas(df.loc[part[p], "calidad"], y_pred, clases, prob)
        if p == "test":
            lo, hi = intervalo_bootstrap(df.loc[part[p], "calidad"], y_pred)
            resultado[p]["f1_macro_ic_inf"], resultado[p]["f1_macro_ic_sup"] = round(lo, 4), round(hi, 4)
            pred["cnn"] = y_pred
            pred[f"cnn_prob_{clases[-1]}"] = prob[:, -1].round(5)
    rutas["resultados"].write_text(json.dumps(resultado, indent=2), encoding="utf8")
    pred.to_csv(rutas["predicciones"], index=False)
    print(json.dumps(resultado, indent=2))
    return resultado
