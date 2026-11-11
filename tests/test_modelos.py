import numpy as np
import pytest

from src.evaluation.evaluar import intervalo_bootstrap, metricas
from src.models.clasicos import crear_modelo, rejilla

CLASES = ["buena", "mala"]


def test_metricas_prediccion_perfecta():
    y = np.array(["buena", "mala", "mala", "buena"])
    m = metricas(y, y, CLASES)
    assert m["exactitud"] == 1.0 and m["f1_macro"] == 1.0 and m["recall_mala"] == 1.0


def test_metricas_auc_y_recall_por_clase():
    y = np.array(["buena"] * 4 + ["mala"] * 4)
    pred = np.array(["buena", "buena", "buena", "mala", "mala", "mala", "mala", "buena"])
    prob = np.array([[.9, .1], [.8, .2], [.7, .3], [.4, .6], [.2, .8], [.1, .9], [.3, .7], [.6, .4]])
    m = metricas(y, pred, CLASES, prob)
    assert m["recall_buena"] == pytest.approx(0.75)
    assert m["recall_mala"] == pytest.approx(0.75)
    assert 0.5 < m["auc"] <= 1.0


def test_metricas_tres_clases():
    clases = ["buena", "regular", "mala"]
    y = np.array(["buena", "regular", "mala"] * 3)
    m = metricas(y, y, clases, np.eye(3)[[0, 1, 2] * 3])
    assert m["f1_regular"] == 1.0 and m["auc"] == 1.0


def test_intervalo_bootstrap_contiene_el_valor():
    rng = np.random.default_rng(0)
    y = rng.choice(CLASES, 300)
    pred = np.where(rng.random(300) < 0.8, y, np.where(y == "buena", "mala", "buena"))
    lo, hi = intervalo_bootstrap(y, pred, n=300)
    assert lo < 0.8 < hi


@pytest.mark.parametrize("nombre", ["logreg", "svm", "rf"])
def test_modelos_clasicos_aprenden_un_problema_separable(nombre):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (60, 4)), rng.normal(3, 1, (60, 4))])
    y = np.array(["buena"] * 60 + ["mala"] * 60)
    modelo = crear_modelo(nombre).fit(X, y)
    assert (modelo.predict(X) == y).mean() > 0.95
    assert all(k.startswith("clf__") for k in rejilla(nombre))


def test_modelo_desconocido():
    with pytest.raises(ValueError):
        crear_modelo("xgb")


def test_cnn_salida_por_clase():
    pytest.importorskip("tensorflow")
    from src.models.cnn import construir_cnn

    modelo = construir_cnn(3, entrada=(64, 64, 3))
    salida = modelo.predict(np.zeros((2, 64, 64, 3), np.float32), verbose=0)
    assert salida.shape == (2, 3)
    assert np.allclose(salida.sum(axis=1), 1.0, atol=1e-5)
