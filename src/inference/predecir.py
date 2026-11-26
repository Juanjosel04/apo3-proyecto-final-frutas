"""Inferencia de punta a punta: de una foto a fruta, calidad y tamaño.

Pasos (los mismos del entrenamiento):
1. Segmentación de la fruta (``mascara_fruta``) y recorte cuadrado con fondo negro.
2. Características manuales del recorte (``extraer_caracteristicas``).
3. Fruta: la que indique el usuario o la del clasificador de fruta (Random Forest).
4. Calidad: SVM ``global_fruta`` (por defecto) o CNN ``doble``.
5. Tamaño relativo con los umbrales por fruta del dataset.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import joblib
import numpy as np
import pandas as pd

from src.config import CHECKPOINTS, CLASES_ENTRENAMIENTO, FRUTAS_PROYECTO, TAM_IMAGEN
from src.data.propias import cargar_umbrales
from src.data.recortes import AREA_REL_MAX, AREA_REL_MIN, recorte_cuadrado
from src.data.segmentacion import mascara_fruta
from src.data.tamano import estimar_tamano
from src.features.extraccion import extraer_caracteristicas

MODELO_SVM = CHECKPOINTS / "clasico_global_fruta_svm.joblib"
MODELO_FRUTA = CHECKPOINTS / "clasificador_fruta_rf.joblib"
MODELO_CNN = CHECKPOINTS / "cnn_calidad_doble.keras"

# Por debajo de esta probabilidad la app sugiere revisión humana
UMBRAL_CONFIANZA = 0.70


@dataclass
class Prediccion:
    fruta: str
    fruta_automatica: bool
    prob_fruta: float | None
    calidad: str
    probabilidades: dict[str, float]
    confianza: float
    tamano: str | None
    diametro_norm: float
    area_rel: float
    segmentacion_ok: bool
    mascara: np.ndarray = field(repr=False)
    recorte: np.ndarray = field(repr=False)
    tiempo_ms: float = 0.0

    @property
    def revisar(self) -> bool:
        """True si la predicción no es confiable y conviene revisarla a mano."""
        return self.confianza < UMBRAL_CONFIANZA or not self.segmentacion_ok


class Clasificador:
    """Carga los modelos una sola vez y predice sobre imágenes BGR."""

    def __init__(self, usar_cnn: bool = False):
        self.svm = joblib.load(MODELO_SVM)
        self.modelo_fruta = joblib.load(MODELO_FRUTA)
        self.umbrales = cargar_umbrales()
        self.cnn = None
        if usar_cnn:
            from tensorflow import keras

            self.cnn = keras.models.load_model(MODELO_CNN)

    def _fruta(self, feats: pd.DataFrame) -> tuple[str, float]:
        prob = self.modelo_fruta.predict_proba(feats)[0]
        i = int(np.argmax(prob))
        return str(self.modelo_fruta.classes_[i]), float(prob[i])

    def _calidad_svm(self, feats: pd.DataFrame, fruta: str) -> np.ndarray:
        x = feats.copy()
        for f in FRUTAS_PROYECTO:
            x[f"fruta_{f}"] = float(f == fruta)
        prob = self.svm.predict_proba(x)[0]
        orden = [list(self.svm.classes_).index(c) for c in CLASES_ENTRENAMIENTO]
        return prob[orden]

    def _calidad_cnn(self, recorte_bgr: np.ndarray) -> np.ndarray:
        x = cv2.resize(recorte_bgr, TAM_IMAGEN, interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32)[None]
        return self.cnn(x, training=False).numpy()[0]

    def predecir(self, img_bgr: np.ndarray, fruta: str | None = None,
                 modelo: str = "svm") -> Prediccion:
        """Predice fruta, calidad y tamaño de una foto con una sola fruta.

        Parameters
        ----------
        img_bgr : np.ndarray
            Foto en BGR.
        fruta : str, opcional
            Fruta conocida. Si es None se usa el clasificador de fruta.
        modelo : {"svm", "cnn"}
            Modelo de calidad.
        """
        t0 = time.perf_counter()
        m = mascara_fruta(img_bgr)
        area_rel = float(np.count_nonzero(m) / m.size)
        seg_ok = AREA_REL_MIN <= area_rel <= AREA_REL_MAX
        if not np.any(m):
            raise ValueError("No se encontró ninguna fruta en la imagen")
        rec, rec_m = recorte_cuadrado(img_bgr, m)
        feats = pd.DataFrame([extraer_caracteristicas(rec, rec_m)])

        automatica = fruta is None
        prob_fruta = None
        if automatica:
            fruta, prob_fruta = self._fruta(feats)
        elif fruta not in FRUTAS_PROYECTO:
            raise ValueError(f"Fruta no soportada: {fruta}")

        if modelo == "cnn":
            if self.cnn is None:
                raise ValueError("El clasificador se creó sin la CNN (usar_cnn=False)")
            prob = self._calidad_cnn(rec)
        elif modelo == "svm":
            prob = self._calidad_svm(feats, fruta)
        else:
            raise ValueError(f"Modelo desconocido: {modelo}")

        i = int(np.argmax(prob))
        t = estimar_tamano(m, self.umbrales[fruta])
        return Prediccion(
            fruta=fruta, fruta_automatica=automatica, prob_fruta=prob_fruta,
            calidad=CLASES_ENTRENAMIENTO[i],
            probabilidades={c: float(p) for c, p in zip(CLASES_ENTRENAMIENTO, prob)},
            confianza=float(prob[i]), tamano=t["categoria"], diametro_norm=t["diametro_norm"],
            area_rel=area_rel, segmentacion_ok=seg_ok, mascara=m, recorte=rec,
            tiempo_ms=(time.perf_counter() - t0) * 1000)
