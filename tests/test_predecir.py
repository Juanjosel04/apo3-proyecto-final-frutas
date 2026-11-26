import cv2
import numpy as np
import pytest

from src.config import FOTOS_PROPIAS
from src.inference.predecir import MODELO_FRUTA, MODELO_SVM, Clasificador

pytestmark = pytest.mark.skipif(not (MODELO_SVM.exists() and MODELO_FRUTA.exists()),
                                reason="faltan los modelos entrenados")


@pytest.fixture(scope="module")
def clasificador():
    return Clasificador()


def _foto():
    ruta = next(FOTOS_PROPIAS.rglob("*.jpg"), None)
    if ruta is None:
        img = np.full((400, 400, 3), 235, np.uint8)
        cv2.circle(img, (200, 200), 90, (30, 60, 200), -1)
        return img
    return cv2.imdecode(np.fromfile(str(ruta), np.uint8), cv2.IMREAD_COLOR)


def test_prediccion_completa(clasificador):
    p = clasificador.predecir(_foto())
    assert p.fruta in ("manzana", "banano", "naranja")
    assert p.calidad in p.probabilidades
    assert sum(p.probabilidades.values()) == pytest.approx(1.0, abs=1e-6)
    assert 0.5 <= p.confianza <= 1.0
    assert p.tamano in ("pequeno", "mediano", "grande")
    assert p.recorte.shape[2] == 3 and p.mascara.shape == _foto().shape[:2]


def test_fruta_indicada_no_usa_el_clasificador(clasificador):
    p = clasificador.predecir(_foto(), fruta="naranja")
    assert p.fruta == "naranja" and not p.fruta_automatica and p.prob_fruta is None


def test_errores_de_entrada(clasificador):
    with pytest.raises(ValueError):
        clasificador.predecir(_foto(), fruta="guayaba")
    with pytest.raises(ValueError):
        clasificador.predecir(_foto(), modelo="xgb")
    with pytest.raises(ValueError):
        clasificador.predecir(_foto(), modelo="cnn")  # se creó sin CNN


def test_revisar_si_la_confianza_es_baja(clasificador):
    p = clasificador.predecir(_foto())
    p.confianza = 0.55
    assert p.revisar
