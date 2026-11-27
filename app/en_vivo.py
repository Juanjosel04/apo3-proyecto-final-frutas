"""Clasificación continua con la cámara (modo en vivo de la app).

Cada cuadro del video llega a ``ProcesadorEnVivo.recv``. La inferencia tarda más
que un cuadro (la segmentación con GrabCut toma unos 100-200 ms), así que corre en
un hilo aparte sobre el cuadro más reciente, y mientras tanto el video sigue
mostrando el contorno y la etiqueta de la última predicción.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from src.inference.predecir import Clasificador, Prediccion

# BGR, mismos tonos que la interfaz
COLOR_BGR = {"buena": (88, 209, 48), "regular": (10, 159, 255), "mala": (58, 69, 255)}
LADO_MAX = 640


class ProcesadorEnVivo(VideoProcessorBase):
    """Procesa el video de la cámara y guarda la última predicción.

    Parameters
    ----------
    clasificador : Clasificador
        Modelos ya cargados (se comparten con el resto de la app).
    intervalo : float
        Tiempo mínimo en segundos entre dos inferencias.
    """

    def __init__(self, clasificador: Clasificador, intervalo: float = 0.3):
        self.clasificador = clasificador
        self.intervalo = intervalo
        self.fruta: str | None = None
        self.modelo = "svm"
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._ocupado = False
        self._t_ultima = 0.0
        self._ultima: Prediccion | None = None
        self._error: str | None = None

    def estado(self) -> tuple[Prediccion | None, str | None]:
        """Última predicción y último error (``None`` si no hay)."""
        with self._lock:
            return self._ultima, self._error

    def _inferir(self, img: np.ndarray) -> None:
        try:
            p, error = self.clasificador.predecir(img, fruta=self.fruta, modelo=self.modelo), None
            if not p.segmentacion_ok:
                p, error = None, "Acerca la fruta y usa un fondo claro y liso"
        except ValueError:
            p, error = None, "Muestra una fruta a la cámara"
        with self._lock:
            self._ultima, self._error, self._ocupado = p, error, False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        escala = LADO_MAX / max(img.shape[:2])
        if escala < 1:
            img = cv2.resize(img, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

        ahora = time.monotonic()
        with self._lock:
            lanzar = not self._ocupado and ahora - self._t_ultima >= self.intervalo
            if lanzar:
                self._ocupado, self._t_ultima = True, ahora
            ultima = self._ultima
        if lanzar:
            self._pool.submit(self._inferir, img.copy())

        if ultima is not None and ultima.mascara.shape == img.shape[:2]:
            img = dibujar(img, ultima)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def dibujar(img: np.ndarray, p: Prediccion) -> np.ndarray:
    """Contorno de la fruta y una etiqueta con la calidad sobre el cuadro."""
    out = img.copy()
    color = COLOR_BGR.get(p.calidad, (255, 255, 255))
    contornos, _ = cv2.findContours(p.mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contornos, -1, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.drawContours(out, contornos, -1, color, 3, cv2.LINE_AA)

    x, y, w, _ = cv2.boundingRect(max(contornos, key=cv2.contourArea))
    texto = f"{p.calidad.upper()}  {p.confianza:.0%}"
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
    x0, y0 = max(0, x + w // 2 - tw // 2 - 12), max(0, y - th - 22)
    capa = out.copy()
    cv2.rectangle(capa, (x0, y0), (x0 + tw + 24, y0 + th + 16), (30, 30, 30), -1)
    out = cv2.addWeighted(capa, 0.65, out, 0.35, 0)
    cv2.putText(out, texto, (x0 + 12, y0 + th + 8), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1, cv2.LINE_AA)
    return out
