import cv2
import numpy as np
import pytest

from src.data.recortes import recorte_cuadrado
from src.features.extraccion import columnas_por_grupo, extraer_caracteristicas


def _disco(lado=200, radio=60, color=(40, 120, 220), centro=None):
    img = np.full((lado, lado, 3), 230, np.uint8)
    m = np.zeros((lado, lado), np.uint8)
    centro = centro or (lado // 2, lado // 2)
    cv2.circle(img, centro, radio, color, -1)
    cv2.circle(m, centro, radio, 255, -1)
    return img, m


def test_recorte_cuadrado_tamano_y_fondo_negro():
    img, m = _disco()
    rec, rec_m = recorte_cuadrado(img, m, lado=96)
    assert rec.shape == (96, 96, 3) and rec_m.shape == (96, 96)
    # el borde de la fruta se mezcla al reducir; lejos de ella el fondo es negro
    lejos = cv2.dilate(rec_m, np.ones((5, 5), np.uint8)) == 0
    assert rec[lejos].max() == 0
    # la fruta ocupa casi todo el cuadrado: pi/4 / (1.12^2) ~ 0.63
    assert 0.55 < np.count_nonzero(rec_m) / rec_m.size < 0.70


def test_recorte_cuadrado_fruta_en_el_borde():
    img, m = _disco(centro=(20, 100))
    rec, rec_m = recorte_cuadrado(img, m, lado=64)
    assert rec.shape == (64, 64, 3)
    assert np.count_nonzero(rec_m) > 0


def test_recorte_mascara_vacia():
    img = np.zeros((50, 50, 3), np.uint8)
    rec, rec_m = recorte_cuadrado(img, np.zeros((50, 50), np.uint8), lado=32)
    assert rec.shape == (32, 32, 3) and rec_m.max() == 0


def test_caracteristicas_numero_y_grupos():
    img, m = _disco()
    f = extraer_caracteristicas(img, m)
    assert len(f) == 78
    assert all(np.isfinite(v) for v in f.values())
    grupos = columnas_por_grupo(f.keys())
    assert {k: len(v) for k, v in grupos.items()} == {
        "hist": 32, "mom": 18, "glcm": 10, "lbp": 10, "forma": 5, "def": 3}


def test_histogramas_suman_uno():
    img, m = _disco()
    f = extraer_caracteristicas(img, m)
    for canal in ("h", "s", "v"):
        total = sum(v for k, v in f.items() if k.startswith(f"hist_{canal}"))
        assert total == pytest.approx(1.0)


def test_circulo_es_circular():
    img, m = _disco()
    f = extraer_caracteristicas(img, m)
    assert f["forma_circularidad"] > 0.85
    assert f["forma_excentricidad"] < 0.2
    assert f["forma_solidez"] > 0.97


def test_fondo_no_cambia_las_caracteristicas():
    img, m = _disco()
    otro = img.copy()
    otro[m == 0] = (0, 255, 0)
    a, b = extraer_caracteristicas(img, m), extraer_caracteristicas(otro, m)
    for k in a:
        if not k.startswith("lbp_"):
            assert a[k] == pytest.approx(b[k], abs=1e-9), k


def test_manchas_oscuras_aumentan_def_oscuro():
    img, m = _disco()
    manchada = img.copy()
    cv2.circle(manchada, (100, 100), 25, (20, 30, 40), -1)
    assert extraer_caracteristicas(manchada, m)["def_oscuro"] > extraer_caracteristicas(img, m)["def_oscuro"] + 0.1


def test_mascara_vacia_lanza_error():
    img, _ = _disco()
    with pytest.raises(ValueError):
        extraer_caracteristicas(img, np.zeros(img.shape[:2], np.uint8))
